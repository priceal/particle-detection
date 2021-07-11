
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 19:37:26 2021

@author: priceal

Evaluates LR, FCN or CNN models.


"""

# define data to evaluate model, must have X and Y values
x_eval =  xtest
y_eval = ytest 

# choose the model to evaluate
model = modelFCN

##############################################################################
##############################################################################

if isinstance(model,linear_model.LogisticRegression):
    x_pred_eval = model.predict( x_eval )
    y_eval_input = y_eval
    x_pred_eval_input = np.round( x_pred_eval )
    
if isinstance(model,torch.nn.modules.container.Sequential):
    x_pred_eval = model( x_eval )
    y_eval_input = y_eval.detach().numpy()
    x_pred_eval_input = np.round( x_pred_eval.detach().numpy() ) 
    
# calculate confusion matrix and eliminate empty axes w/ squeeze
cf = confusion_matrix( 
                        np.squeeze( y_eval_input ), \
                        np.squeeze( x_pred_eval_input )
                     )
                        
truePositives = cf[1,1]
falsePositives = cf[0,1]
trueNegatives = cf[0,0]
falseNegatives = cf[1,0]
negatives = trueNegatives + falseNegatives
positives = truePositives + falsePositives
actualHits = truePositives + falseNegatives
actualMisses = trueNegatives + falsePositives

print('')
print('test set contains {:d} TRUE and {:d} FALSE'.format(actualHits,actualMisses))
print('predictions contain {} POSITIVE and {} NEGATIVE'.format(positives,negatives))
print('')
print('recall    {:>2.1f}%'.format(100*truePositives/actualHits) )
print('precision {:>2.1f}%'.format(100*truePositives/(truePositives+falsePositives)) )
frmt = '{:<10} {:>10} {:>10} {:>10}'
print('')
print('           **CONFUSION MATRIX**')
print(frmt.format(' ','FALSE','TRUE','TOTAL'))
print(frmt.format('NEGATIVE',trueNegatives,falseNegatives,negatives))
print(frmt.format('POSITIVE',falsePositives,truePositives,positives))
print(frmt.format('TOTAL',actualMisses,actualHits,len(y_eval)))

