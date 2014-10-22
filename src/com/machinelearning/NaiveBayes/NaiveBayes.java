package com.machinelearning.NaiveBayes;

import java.util.ArrayList;
import java.util.HashMap;

public class NaiveBayes {
	public static ArrayList<Double> weights;
	
	public static void main(String[] args) {
		
		//Read the file name for training data from config file
		String trainFilePath = Config.readConfig("trainFileName");
		String testFilePath = Config.readConfig("testFileName");
		
		ArrayList<Example> examples = DataLoader.readRecords(trainFilePath);
		System.out.println("Examples =" + examples.size());
		System.out.println("Features : "+DataLoader.featurePossVals);
		
		HashMap<Integer, HashMap<String, Integer>> trueCounts = new HashMap<Integer, HashMap<String, Integer>>();
		HashMap<Integer, HashMap<String, Integer>> falseCounts = new HashMap<Integer, HashMap<String, Integer>>();
		
		
		int lableIndex = DataLoader.labels.indexOf("type");
		String trueClassLable = "1";
		int trueLableExamples = 0;
		String currVal = null;
		for(Example ex : examples){
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
		int possClassLables = 2;
		
		System.out.println("True HashMap : "+trueCounts.toString());
		System.out.println("False HashMap : "+falseCounts.toString());
		
		//Classify
		ArrayList<Example> testExamples = DataLoader.readRecords(testFilePath);
		System.out.println("Test Examples =" + testExamples.size());
		
		int correctPredctionCount = 0;
		int wrongPredctionCount = 0;
		
		for(Example testEx : testExamples){
			String observedLable = testEx.getFeature(lableIndex);
			
			double trueClassPrior = (double)(trueLableExamples + 1) / (trueLableExamples + falseLableExamples + possClassLables);
			double falseClassPrior = (double)(falseLableExamples + 1) / (trueLableExamples + falseLableExamples + possClassLables);
			double trueProb = 1.0 * trueClassPrior;
			double falseProb = 1.0 * falseClassPrior;
			
			for(int i = 0; i < DataLoader.numberOfFeatures; i++){
				String featureVal = testEx.getFeature(i);
				int numberOfPossVals = DataLoader.featurePossVals.get(i).size();
				if(i != lableIndex){
					//for unseen values of features in train data assigning count = 0
					int featureTrueCount = trueCounts.get(i).get(featureVal) != null ? trueCounts.get(i).get(featureVal) : 0;
					int featureFalseCount = falseCounts.get(i).get(featureVal) != null ? falseCounts.get(i).get(featureVal) : 0;
					
					trueProb *= (double)(featureTrueCount + 1)/(trueLableExamples + numberOfPossVals);
					//System.out.println("FFFFFF : "+i+" "+testEx.getFeature(i));
					falseProb *= (double)(featureFalseCount + 1)/(falseLableExamples + numberOfPossVals);
					
					System.out.println("true prob : "+(featureTrueCount + 1)+"/"+trueLableExamples+"+"+numberOfPossVals);
					System.out.println("true prob : "+(featureFalseCount + 1)+"/"+falseLableExamples+"+"+numberOfPossVals);
				}
			}			
			System.out.println("=========================");
			
			double normTrueProb = trueProb / (trueProb + falseProb);
			double normFalseProb = falseProb / (trueProb + falseProb);
			
			String prediction = normTrueProb >= normFalseProb ? "T" : "F";
			
			if(observedLable.equalsIgnoreCase("1")){
				if(!prediction.equalsIgnoreCase("T")){
					wrongPredctionCount++;
				}
			}
			else{
				if(prediction.equalsIgnoreCase("T")){
					wrongPredctionCount++;
				}
			}
		}
		
		System.out.println(wrongPredctionCount+" Incorrect Predictions for "+testExamples.size()+" test examples");
		
		//for(int i = 0; i<)
		
		
		//DEBUG
		/*for (int i =0; i<examples.size();i++){
			for (int j =0; j<DataLoader.numberOfFeatures;j++){
				
				System.out.print(examples.get(i).getFeature(j));
				
			}
			System.out.println();
		}*/
	}

}
