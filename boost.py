import numpy as np

class boosting:
#we convert to list the array numpy
    def __init__(self,max_depth,min_size,num_trees):
        self.max_depth = max_depth
        self.min_size = min_size
        self.num_trees = num_trees
        self.lst_dt = None
        self.alphas = []
    def train_boost(self,X,Y):
        ep = 10**-6
        et_arr = []
        lst_dt = []
        #sz = int(dataset.shape[0]*self.sz_pro)
        wt = np.ones(shape = Y.shape)
        wt = wt/np.sum(wt)
        wt_y = np.multiply(wt,Y)
        dataset =  np.hstack((X,wt_y))
        for i in range(self.num_trees):
            #print('on {} tree'.format(i+1))
		#the tree gonaaa remain same, just count negatives instead of 0 and positives instead of 1
            d_t = decision_tree_boost(max_depth=self.max_depth,min_size=self.min_size)
            d_t.train(dataset.tolist())
            lst_dt.append(d_t)
            
            y_predict = d_t.predict(np.delete(dataset,-1,axis=1))
            y_predict = np.array(y_predict).reshape(Y.shape)
            
            et = np.sum(wt[Y !=y_predict])
            et_arr.append(et)
            if et == 0:
                alpha = (0.5)*np.log(((1 - et + ep)/(et + ep)))
                self.alphas.append(alpha)
                break
            Y_Y_pred = np.multiply(Y,y_predict)
            alpha = 0.5*np.log((1-et)/et)
            self.alphas.append(alpha)
            
            wt = np.multiply(wt,np.exp(-1.0*alpha*Y_Y_pred))
            #print(np.sum(wt))
            wt = wt/np.sum(wt)
           # print(np.sum(wt))
            
            reshape_shape = dataset[:,-1].shape
            dataset[:,-1] = np.multiply(wt,Y).reshape(reshape_shape)         
            
        self.lst_dt = lst_dt
        return et_arr
        
    def test_boost(self,X):
        pred_y_arr = []
        i=0
        for dt in self.lst_dt:
            pred_y = dt.predict(X.tolist())
            #print(max(pred_y))
            #print(min(pred_y))
            pred_y_arr.append(self.alphas[i]*np.array(pred_y))
            i+=1
            #print('testing {}'.format(i))
        return pred_y_arr

    
def boosting_function(X_train,Y_train,X_test,Y_test,max_depth,min_size,num_trees):
    Y_train[Y_train==0] = -1.0
    Y_train[Y_train==1] = 1.0
    Y_test[Y_test==0] = -1.0
    Y_test[Y_test==1] = 1.0   
    
    Y_train = Y_train.reshape((X_train.shape[0],1))
    Y_test = Y_test.reshape((X_test.shape[0],1))
    
    boost_t= boosting(max_depth=max_depth,min_size=min_size,num_trees=num_trees)
    et_arr = boost_t.train_boost(X_train,Y_train)

    pred_y_arr = np.array(boost_t.test_boost(X_test))

    y_test_predict = np.sum(pred_y_arr,axis=0)
    y_test_predict[y_test_predict>=0]=1
    y_test_predict[y_test_predict<0]=-1
    err = error(list(Y_test[:,0]),list(y_test_predict))
    return err
    
    
    