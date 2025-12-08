class CustomLoss(nn.Module):
    """
    Custom loss function
    sum_n ||  ( I-PP^+(fn[Uμ]) )v_lambda ||₂
    The class is defined as subclass of nn.Module
    """
    def __init__(self,pred_test_vectors,test_vectors):
        super(CustomLoss, self).__init__() #Calls the constructor of the base class, i.e. Module

        self.test_vectors = test_vectors           #[batch_size,NV,2,NT,NX]
        self.pred_test_vectors = pred_test_vectors #[batch_size,NV,2,NT,NX]
        self.batch_size = self.test_vectors.shape[0]
    
    def forward(self):
        #P_Pdagg(f_n) = P(f_n)P^+(f_n)f_n
        loss = 0
        for i in range(self.batch_size):
            ops = op.Operators(var.BLOCKS_X,var.BLOCKS_T,self.pred_test_vectors[i])
            for tv in range(var.NV):
                #Evaluate 
                loss += np.linalg.norm( (self.test_vectors[i,tv] - ops.P_Pdagg(self.test_vectors[i,tv])) )
            #loss = np.linealg.norm(test_vectors - P_Pdagg(test_vectors)) #|| . ||₂ (l2-norm)
            #We normalize the loss function, otherwise the minimization is unstable.
        return loss/(self.batch_size*var.NV)