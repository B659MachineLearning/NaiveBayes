package com.machinelearning.NaiveBayes;

/*
 * Authors : Aniket Bhosale and Mayur Tare
 * Description : This class implements Naive Bayes algorithm with Laplace correction.
 */

import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayes {
	public static ArrayList<Double> weights;
	//Assume class label "1" to be true and rest to be false
	public static String trueClassLable = "1";
	
	
	public static void main(String[] args) {
		
		//Read the file name for training data from config file
		String trainFilePath = Config.readConfig("trainFileName");
		String testFilePath = Config.readConfig("testFileName");
		
		//Read Training Examples from Dataset file
		ArrayList<Example> examples = DataLoader.readRecords(trainFilePath);
		System.out.println("Examples =" + examples.size());
		
		//Data Structure to store the counts for features
		HashMap<Integer, HashMap<String, Integer>> trueCounts = new HashMap<Integer, HashMap<String, Integer>>();
		HashMap<Integer, HashMap<String, Integer>> falseCounts = new HashMap<Integer, HashMap<String, Integer>>();
		
		//Index of the class label
		String lable = Config.readConfig("classLable");
		int lableIndex = DataLoader.labels.indexOf(lable);
		
		int trueLableExamples = 0;
		String currVal = null;
		
		//Iterate over all training examples and learn.
		for(Example ex : examples){
			//Count for true class lable
			if(ex.features.get(lableIndex).equalsIgnoreCase(trueClassLable)){
				trueLableExamples++;
				for(int i = 0; i < DataLoader.numberOfFeatures; i++){
					currVal = ex.features.get(i);
					
					if(!trueCounts.containsKey(i)){
						HashMap<String, Integer> featureMap = new HashMap<String, Integer>();
						featureMap.put(currVal, 1);
						trueCounts.put(i, featureMap);
					}
					else{
						if(!trueCounts.get(i).containsKey(currVal)){
							trueCounts.get(i).put(currVal, 1);
						}
						else{
							int currCount = trueCounts.get(i).get(currVal);
							trueCounts.get(i).put(currVal, currCount+1);
						}
					}					
				}
			}
			//Count for false class label
			else{
				for(int i = 0; i < DataLoader.numberOfFeatures; i++){
					currVal = ex.features.get(i);
					
					if(!falseCounts.containsKey(i)){
						HashMap<String, Integer> featureMap = new HashMap<String, Integer>();
						featureMap.put(currVal, 1);
						falseCounts.put(i, featureMap);
					}
					else{
						if(!falseCounts.get(i).containsKey(currVal)){
							falseCounts.get(i).put(currVal, 1);
						}
						else{
							int currCount = falseCounts.get(i).get(currVal);
							falseCounts.get(i).put(currVal, currCount+1);
						}
					}					
				}
			}
		}
		//Total number of False lable Examples
		int falseLableExamples = examples.size() - trueLableExamples;
		
		//Possible class lables
		int possClassLables = 2;

		
		//Classify the test data
		//Read Test examples from Test Dataset
		ArrayList<Example> testExamples = DataLoader.readRecords(testFilePath);
		System.out.println("Test Examples =" + testExamples.size());
		
		int wrongPredctionCount = 0;
		int laplaceCorrection = 1;
		
		for(Example testEx : testExamples){
			String observedLable = testEx.getFeature(lableIndex);
			
			double trueClassPrior = (double)(trueLableExamples + laplaceCorrection) / (trueLableExamples + falseLableExamples + possClassLables);
			double falseClassPrior = (double)(falseLableExamples + laplaceCorrection) / (trueLableExamples + falseLableExamples + possClassLables);
			double trueProb = 1.0 * trueClassPrior;
			double falseProb = 1.0 * falseClassPrior;
			
			for(int i = 0; i < DataLoader.numberOfFeatures; i++){
				String featureVal = testEx.getFeature(i);
				int numberOfPossVals = DataLoader.featurePossVals.get(i).size();
				if(i != lableIndex){
					//for unseen values of features in train data assigning count = 0
					int featureTrueCount = 0;
					int featureFalseCount = 0;
					if(trueCounts != null && trueCounts.get(i) != null)
						featureTrueCount = trueCounts.get(i).get(featureVal) != null ? trueCounts.get(i).get(featureVal) : 0;
					if(falseCounts != null && falseCounts.get(i) != null)	
						featureFalseCount = falseCounts.get(i).get(featureVal) != null ? falseCounts.get(i).get(featureVal) : 0;
					
					trueProb *= (double)(featureTrueCount + laplaceCorrection)/(trueLableExamples + numberOfPossVals);
					falseProb *= (double)(featureFalseCount + laplaceCorrection)/(falseLableExamples + numberOfPossVals);
					
				}
			}
						
			double normTrueProb = trueProb / (trueProb + falseProb);
			double normFalseProb = falseProb / (trueProb + falseProb);
			
			String prediction = normTrueProb >= normFalseProb ? "T" : "F";
			
			//Check if prediction is correct or not
			if(observedLable.equalsIgnoreCase(trueClassLable)){
				if(!prediction.equalsIgnoreCase("T")){
					wrongPredctionCount++;
					System.out.println("Wrong prediction for Test Example : "+testEx.features.toString()+" Predicted "+prediction);
				}
			}
			else{
				if(prediction.equalsIgnoreCase("T")){
					wrongPredctionCount++;
					System.out.println("Wrong prediction for Test Example : "+testEx.features.toString()+" Predicted "+prediction);
				}
			}
		}
		
		//Print the report of the classification
		System.out.println(wrongPredctionCount+" Incorrect Predictions for "+testExamples.size()+" test examples");
	}

}
