import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np    

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
data_dict=pickle.load(open("data.pickle","rb"))
# print(data_dict.keys())
# print(data_dict)

data=np.asarray(data_dict["data"])
labels=np.asarray(data_dict["labels"])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,shuffle=True,stratify=labels)
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# for i in range(3):
#     tree = rf.estimators_[i]
#     dot_data = export_graphviz(tree,
#                                feature_names=X_train.columns,  
#                                filled=True,  
#                                max_depth=2, 
#                                impurity=False, 
#                                proportion=True)
#     graph = graphviz.Source(dot_data)
#     display(graph)

f=open("model.pickle","wb")
pickle.dump({"model":model},f)
f.close()
