# Wikiled.MachineLearning.Svm
Native C# multithreaded version of LIBSVM


Original code is taken from one of LIBSVM ports, fixed some small bugs + added multithreading for better performance

Simple training example
```c#
SvmTrainClient train = new SvmTrainClient(arff);
var model = await train.Train(header, token);
```

Full training example
```c#
var arff = ArffDataSet.CreateSimple("subjectivity");
arff.Header.RegisterNominalClass(labels);
foreach (var definition in documentSet.Document.Where(item => item.Labels.Length > 0))
{					
	var label = definition.Labels.Last();				
	var review = dataHolder.AddReview();
	review.Class.Value = label;
	foreach (var record in definition.WordsTable)
	{
		review.AddRecord(record.Key).Value = record.Value;
	}					
}

arff.CompactHeader(3);
arff.CompactReviews(3);
arff.CompactClass(3);

if (arff.Reviews.Length < 10)
{
	throw new LearningException("Not enough documents to learn patterns");
}

arff.Normalize(header.Normalization);
SvmTrainClient train = new SvmTrainClient(arff);
var model = await train.Train(header, token);
```


Grid search to find best parameters:
```c#
var training = new TrainingModel();
var taskFactory = new TaskFactory(new LimitedConcurrencyLevelTaskScheduler(2));
var parameters = new GridSearchParameters(5, new double[] { 1, 2, 3, 4 }, new double[] { 1, 2, 3, 4 }, new Parameter());
var instance = new GridParameterSelection(taskFactory, training, parameters);
var dataSet = ArffDataSet.CreateSimple("Test");
dataSet.Header.RegisterNominalClass("One", "Two", "Three");
dataSet.UseTotal = true;
var one = dataSet.AddDocument();
one.Class.Value = "One";
one.AddRecord("Good");
problem = dataSet.GetProblem();
var result = await instance.Find(problem, CancellationToken.None); result.
item.AddRecord("a");
```


  
