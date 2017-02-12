'''
Developer: Juan Carcamo
Purpose: Class to hold and access classifier information. 

Details: 

Credits: Juan Gonzalo Carcamo
'''

class Classifier:
    
    def __init__(self):
        self.name = ''
        self.class_probability = 0.0
        self.features = {}

    def print_classifier(self):
        print ("Name %s"%(self.name))
        print ("Class Probability %.10f"%(self.class_probability))
        print ("Number of features %d"%(len(self.features)))

    def get_feature_probability(self,feature):
        if feature in self.features:
            return self.features[feature]
        else:
            return 0
