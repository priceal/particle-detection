
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
model = modelCNN

##############################################################################
##############################################################################

if isinstance(model,linear_model.LogisticRegression):
    x_pred_eval_input = model.predict( x_eval ).astype(np.uint8)
    x_pred_prob = model.predict_proba( x_eval )[:,1]
    y_eval_input = y_eval
    
if isinstance(model,torch.nn.modules.container.Sequential):
    x_pred_prob = model( x_eval ).detach().numpy()
    x_pred_eval_input = np.round( x_pred_prob ).astype(np.uint8)
    y_eval_input = y_eval.detach().numpy()
    
# calculate confusion matrix and eliminate empty axes w/ squeeze
y_true = np.squeeze( y_eval_input )
y_pred = np.squeeze( x_pred_eval_input )
y_prob = np.squeeze( x_pred_prob )
cf = confusion_matrix( y_true, y_pred )
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
print('       **CONFUSION MATRIX (THRESHOLD=0.5)**')
print(frmt.format(' ','FALSE','TRUE','TOTAL'))
print(frmt.format('NEGATIVE',trueNegatives,falseNegatives,negatives))
print(frmt.format('POSITIVE',falsePositives,truePositives,positives))
print(frmt.format('TOTAL',actualMisses,actualHits,len(y_eval)))

prc = precision_recall_curve(y_true,y_prob)
plt.plot(prc[1],prc[0])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('precision-recall curve')