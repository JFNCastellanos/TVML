import numpy as np
import torch
import torch.optim as optim
import torch.nn.parallel
import parameters as var #Configuration and coarsening parameters
var.init() #initializes parameter
import loss_function as lf #Custom loss function


def train(dataloader, model, optimizer,losses,version):
    """
    Trains the model based on the test loader
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader that yields the training loader:
        (confs_batch, near_kernel)
    model : torch.nn.Module
        The model to evaluate.
    Appends loss to losses (list declared outside the function)
    version: 
        0 -> Assemble P, P^+ with SAP test vectors and find other vectors similar to them.
        1 -> Assemble P, P^+ with the predicted test vectors and verify that SAP vectors are in the image.
    """
    model.train()
    criterion = lf.CustomLossTorch().to(var.DEVICE)               # instantiate once, reuse
    for batch_id, batch in enumerate(dataloader):
        # -------------------------------------------------
        # Load the data
        # -------------------------------------------------
        confs_batch   = batch[0].to(var.DEVICE)            # shape (B, …)
        near_kernel   = batch[1].to(var.DEVICE)            # shape (B, NV, 2, NT, NX)

        # -------------------------------------------------
        # Forward pass of the model → predicted test vectors
        # -------------------------------------------------
        # model returns a real‑valued tensor of shape [B, 4*NV, NT, NX]
        pred = model(confs_batch)                     # still a torch Tensor
        # -------------------------------------------------
        # Reshape / convert to complex dtype
        # -------------------------------------------------
        # Example: real/imag in 4 channels (Re0, Re1, Im0, Im1)
        B = pred.shape[0]                         #Batch size
        pred = pred.view(B, var.NV, 4, var.NT, var.NX)      # (B,NV,4,NT,NX)

        # Build a complex tensor of shape (B, NV, 2, NT, NX)
        #   channel 0 → real part of component 0
        #   channel 1 → real part of component 1
        #   channel 2 → imag part of component 0
        #   channel 3 → imag part of component 1
        real = torch.stack([pred[:,:, 0], pred[:,:, 1]], dim=2)   # (B,NV,2,NT,NX)
        imag = torch.stack([pred[:,:, 2], pred[:,:, 3]], dim=2)   # (B,NV,2,NT,NX)
        pred_complex = torch.complex(real, imag)                  # (B,NV,2,NT,NX)

        if version == 0:
            #Normalizing the fake test vectors
            norms = torch.linalg.vector_norm(pred_complex[:,:],dim=(-3,-2, -1)).view(B, var.NV, 1, 1, 1)
            pred_complex_normalized = pred_complex / norms
            #We assemble P, P^+ with the SAP test vectors and find other vectors that are similar to them.
            loss = criterion(near_kernel, pred_complex_normalized)   # loss is a scalar Tensor
        elif version == 1:
            norms = torch.linalg.vector_norm(near_kernel[:,:],dim=(-3,-2,-1)).view(B,var.NV,1,1,1)
            near_kernel_normalized = near_kernel / norms
            #We assemble P, P^+ with the predicted test vectors and verify that SAP vectors are in the image.
            loss = criterion(pred_complex, near_kernel_normalized)   # loss is a scalar Tensor
        else:
            print("Choose a valid version")
            return None
       
        # -------------------------------------------------
        # Back‑propagation
        # -------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------
        loss_val = loss.item()
        current = (batch_id + 1) * B
        losses.append(loss_val)
        print(f"loss: {loss_val:>7f}  [{current:>5d}/{var.TRAIN_LEN:>5d}]") 



def evaluate(dataloader, model, device, version,criterion=None):
    """
    Run a forward pass on the whole test set and return
    the mean loss.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader that yields the same tuple format as the training loader:
        (confs_batch, near_kernel)
    model : torch.nn.Module
        The model to evaluate. It will be switched to ``eval`` mode for the
        duration of the call and restored to its previous mode afterwards.
    device : torch.device
        Where the tensors should live (e.g. ``torch.device('cuda')``).
    criterion : torch.nn.Module, optional
        A loss object that implements ``__call__(target, prediction)``.
        If ``None`` a fresh ``lf.CustomLossTorch`` instance will be created
        on the supplied ``device`` (the same as in the training loop).
    version: 
        0 -> Assemble P, P^+ with SAP test vectors and find other vectors similar to them.
        1 -> Assemble P, P^+ with the predicted test vectors and verify that SAP vectors are in the image.

    Returns
    -------
    avg_loss : float
        Mean loss over all batches (scalar).
    batch_losses : list[float]
        Individual batch losses.
    """
    # ------------------------------------------------------------------
    # Put the model in eval mode – turns off dropout, batch‑norm updates
    # ----------------------------------------------------------------
    was_training = model.training          # remember the original state
    model.eval()

    # ------------------------------------------------------------------
    # Build (or reuse) the loss object on the proper device
    # ------------------------------------------------------------------
    if criterion is None:
        criterion = lf.CustomLossTorch().to(device)

    # ------------------------------------------------------------------
    # Containers for the per‑batch losses
    # ------------------------------------------------------------------
    batch_losses = []

    # ------------------------------------------------------------------
    # No gradient tracking – saves memory and speeds up the forward pass
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):
            # -------------------------------------------------
            # Load the data (move to the same device as the model)
            # -------------------------------------------------
            confs_batch = batch[0].to(device)          # (B, …)
            near_kernel = batch[1].to(device)          # (B, NV, 2, NT, NX)

            # -------------------------------------------------
            # Forward pass – exactly the same reshaping as in training
            # -------------------------------------------------
            pred = model(confs_batch)                  # (B, 4*NV, NT, NX)

            B = pred.shape[0]    #Batch size
            pred = pred.view(B, var.NV, 4, var.NT, var.NX)   # (B,NV,4,NT,NX)

            # Build complex tensor (B,NV,2,NT,NX)
            real = torch.stack([pred[:, :, 0], pred[:, :, 1]], dim=2)   # (B,NV,2,NT,NX)
            imag = torch.stack([pred[:, :, 2], pred[:, :, 3]], dim=2)   # (B,NV,2,NT,NX)
            pred_complex = torch.complex(real, imag)
            if version == 0:
                #Normalizing the fake test vectors
                norms = torch.linalg.vector_norm(pred_complex[:,:],dim=(-3,-2, -1)).view(B, var.NV, 1, 1, 1)
                pred_complex_normalized = pred_complex / norms
                #We assemble P, P^+ with the SAP test vectors and find other vectors that are similar to them.
                loss = criterion(near_kernel, pred_complex_normalized)   # loss is a scalar Tensor
            elif version == 1:
                norms = torch.linalg.vector_norm(near_kernel[:,:],dim=(-3,-2,-1)).view(B,var.NV,1,1,1)
                near_kernel_normalized = near_kernel / norms
                #We assemble P, P^+ with the predicted test vectors and verify that SAP vectors are in the image.
                loss = criterion(pred_complex, near_kernel_normalized)   # loss is a scalar Tensor
            else:
                print("Choose a valid version")
                return None
            
            # -------------------------------------------------
            # Store loss value (as a Python float)
            # -------------------------------------------------
            loss_val = loss.item()
            batch_losses.append(loss_val)

    # ------------------------------------------------------------------
    # Compute the mean loss over the whole set
    # ------------------------------------------------------------------
    batch_losses = np.array(batch_losses)
    avg_loss = np.mean(batch_losses) 
    std_err = np.std(batch_losses)/np.sqrt(len(batch_losses))

    # ------------------------------------------------------------------
    # Restore the original training/eval state of the model
    # ------------------------------------------------------------------
    model.train(was_training)

    return avg_loss, std_err, batch_losses