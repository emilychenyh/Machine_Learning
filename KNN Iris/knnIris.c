// Emily Chen, November 18, 2016
// KNN algo from scratch to sort Iris data
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <math.h>

// File defines
#define MAX_CHARS 40	// max # of chars in each line of input in data file (actually 31)
#define FILE_PATH "Iris_data_set/iris.data.txt"

// Guesses for max file sizes (do know the max, but this is good practice)
#define MAX_DATA 160	// data file has 150 lines
#define INCREASE 10		// number of array slots to increase reallocs by
#define MAX_FIELD 20	// max char size of any field that's read in (actual = 15)

// Classes of Iris (internal use)
#define IRIS_SETOSA 0
#define IRIS_VERSICOLOR 1
#define IRIS_VIRGINICA 2
#define NUM_CLASSES 3

// Classes as given in the data file
#define SETOSA "Iris-setosa"
#define VERSICOLOR "Iris-versicolor"
#define VIRGINICA "Iris-virginica"

// Settings for KNN (change into user input later)
#define SPLIT 66	// percent of data to use as training data (other is test), 
					// 67/33 for train/test standard
#define K_CHOICE 3	// the value of k for the k nearest neighbors we want to find

// What each line of the csv file holds
#define NUM_FIELDS 5	// number of fields in Iris struct
typedef struct _Iris {
	double sepLen;	// Sepal Length
	double sepWidth;// Sepal Width
	double petLen;	// Petal Length
	double petWidth;// Petal Width
	int class;		// see classes of iris internal defines
} Iris;

// for computing neighbors
typedef struct _IrisSort {
	int dist;	// distance from an instance
	Iris *item;	// pointer to its Iris item
} IrisSort;

// read from the file object and split the data into the two Iris arrays
void loadData (FILE *f, int split, Iris *training, Iris *test, int* trainIx, int* testIx);
void storeLine (char* line, Iris* set, int index);	// store a line of input into either train/test set
void checkAvailability (Iris* set, int index, int *numRealloc);	// checks if reallocs are needed
void testLoadData (Iris *training, Iris *test, int trainSize, int testSize);

// Euclidian distance function to calculate similarity between data points
double euclidDist (Iris data1, Iris data2);
void testEuclidDist (void);

// Calculate the k most similar instances (neighbors) and return it in an array
Iris *getNeighbors (Iris* training, int trainSize, Iris testInstance, int k);
void selectionSort(IrisSort a[], int lo, int hi);
void testGetNeighbors (void);

// Calculate a predicted response based on neighbors
int getResponse(Iris* neighbors, int k);
//void selectionSortInt(int a[], int lo, int hi);
void testGetResponse(void);

// accuracy of predictions as a percent of correct classifications
double getAccuracy (Iris *test, int testSize, int* predictions);
void testGetAccuracy(void);

// debugging
void showIrisArray (Iris *a, int n);
// print detailed results
// modes: 0 = all results, 1 = only results that differ
void printResults(Iris* test, int testSize, int* predictions, int mode);

int main (int argc, char* argv[]) {

	// === Get split ratio and k value ===

	// get the settings for KNN from command line, implement last ----------------------------
	// int split, kChoice;
	int kchoice = K_CHOICE;
	int split = SPLIT;

	// === Prepare the Data ===

	// open up the file
	FILE *data;
	data = fopen(FILE_PATH, "r");	// open up for reading
	assert(data != NULL);			// successful file opening

	// load data into Iris arrays
	Iris *training = malloc(MAX_DATA * sizeof(Iris));
	assert(training != NULL);
	//printf("sizeof(training) = %zu\n", sizeof(training));
	Iris *test = malloc(MAX_DATA * sizeof(Iris));
	assert(test != NULL);
	//printf("sizeof(test) = %zu\n", sizeof(test));
	int trainSize, testSize;

	loadData(data, split, training, test, &trainSize, &testSize);

	fclose(data);

	//testLoadData(training, test, trainSize, testSize);
	//testEuclidDist();
	//testGetNeighbors();
	//testGetResponse();
	//testGetAccuracy();

	// === Generate Predictions For Test Array ===
	int *predictions = malloc(testSize * sizeof(int));
	assert (predictions != NULL);
	int i, class;
	Iris *neighbors;
	for (i = 0; i < testSize; i++) {
		// get neighbors for current test Iris
		neighbors = getNeighbors(training, trainSize, test[i], kchoice);
		// get the majority class vote
		class = getResponse(neighbors, kchoice);
		predictions[i] = class;
		
		// free malloced area, prepare for next iteration
		free(neighbors);
		neighbors = NULL;
	}

	// === Get Accuracy ===
	// accuracy should be around 98%, implementation problems?
	// stability of sorts?
	double accuracy = getAccuracy(test, testSize, predictions);

	// === Print results to screen ===
	printf("Accuracy = %lf%%\n", accuracy);
	printResults(test, testSize, predictions, 1);

	// remember to free all arrays and mallocs
	free(training);
	free(test);
	free(predictions);

	return EXIT_SUCCESS;
}

// print detailed results
// modes: 0 = all results, 1 = only results that differ
void printResults(Iris* test, int testSize, int* predictions, int mode) {
	int i;
	switch (mode) {
	case 0:
		// all results
		printf("Printing all results\n");
		for (i = 0; i < testSize; i++) {
			printf("test[%d].class = %d\tpred[%d] = %d\n", i, test[i].class, i, predictions[i]);
		}
		break;
	case 1:
		// only results that differ
		printf("Printing results that differ\n");
		for (i = 0; i < testSize; i++) {
			if(test[i].class != predictions[i]) {
				printf("test[%d].class = %d\tpred[%d] = %d\n", i, test[i].class, i, predictions[i]);	
			}
		}
		break;
	default:
		// all results
		printf("Printing all results\n");
		for (i = 0; i < testSize; i++) {
			printf("test[%d].class = %d\tpred[%d] = %d\n", i, test[i].class, i, predictions[i]);
		}
		break;
	}
	
}

// simple test function for getAccuracy()
void testGetAccuracy(void) {
	printf("Testing getAccuracy()...\n");

	Iris new1 = {5.1, 3.5, 1.4, 0.2, 0};
	Iris new2 = {7.0, 3.2, 4.7, 1.4, 1};
	Iris new3 = {6.9, 3.1, 5.4, 2.1, 2};
	Iris base1 = {1, 2, 3, 4, 0};
	Iris base2 = {1, 2, 3, 4, 1};

	int testSize;

	// 100% accuracy on 1 item
	printf("== Test 1 ==");
	Iris *test1 = &new3;
	testSize = 1;
	int preds1[testSize];
	preds1[0] = 2;
	printf("Should return 100%%\n");
	printf("result = %lf%%\n", getAccuracy(test1, testSize, preds1));

	// 100% accuracy on 2 items
	printf("== Test 2 ==");
	testSize = 2;
	Iris test2[testSize];
	test2[0] = new2;
	test2[1] = base2;
	int preds2[2] = {1, 1};
	printf("Should return 100%%\n");
	printf("result = %lf%%\n", getAccuracy(test2, testSize, preds2));

	// 100% accuracy on 5 items
	printf("== Test 3 ==");
	testSize = 5;
	Iris test3[5] = {new1, new2, new3, base1, base2};
	int preds3[5] = {0, 1, 2, 0, 1};
	printf("Should return 100%%\n");
	printf("result = %lf%%\n", getAccuracy(test3, testSize, preds3));

	// 2/3 accuracy on 3 items
	printf("== Test 4 ==");
	testSize = 3;
	Iris test4[3] = {new1, new2, new3};
	int preds4[3] = {0, 1, 1};
	printf("Should return 66.67%%\n");
	printf("result = %lf%%\n", getAccuracy(test4, testSize, preds4));

	// 1/2 accuracy on 2 items
	printf("== Test 5 ==");
	testSize = 2;
	Iris test5[2] = {base1, base2};
	int preds5[2] = {0, 0};
	printf("Should return 50%%\n");
	printf("result = %lf%%\n", getAccuracy(test5, testSize, preds5));

	// 0% accuracy on 5 items.
	printf("== Test 6 ==");
	testSize = 5;
	Iris test6[5] = {new1, new2, new3, base1, base2};
	int preds6[5] = {1, 2, 0, 2, 0};
	printf("Should return 0%%\n");
	printf("result = %lf%%\n", getAccuracy(test6, testSize, preds6));

	printf("All tests printed. Do a visual check (float are annoying).\n");
}

// accuracy of predictions as a percent of correct classifications
// predictions array is same size as testSize
double getAccuracy (Iris *test, int testSize, int* predictions) {
	int correct = 0;
	int i;

	// loop through and see if prediction for that test Iris is correct
	for (i = 0; i < testSize; i++) {
		if (predictions[i] == test[i].class) {
			correct++;
		}
	}

	return (correct/(double)(testSize)) * 100.0;
}

// simple test function for getResponse()
void testGetResponse(void) {
	printf("Testing getResponse()...\n");

	Iris new1 = {5.1, 3.5, 1.4, 0.2, 0};
	Iris new2 = {7.0, 3.2, 4.7, 1.4, 1};
	Iris new3 = {6.9, 3.1, 5.4, 2.1, 2};
	Iris base1 = {1, 2, 3, 4, 0};
	Iris base2 = {1, 2, 3, 4, 1};

	// what to do when # votes is the same? depends on the stability of the sort
	// although with real data this probably wouldn't happen and you can try to have an odd k
	// or take an unbiased random response

	int k;	// chosen k value
	//int response;

	// class vote of 1 item is 1 item
	printf("== Test 1 ==\n");
	Iris *neighbors1 = &new1;
	k = 1;
	//response = getResponse(neighbors1, k);
	//printf("getResponse(neighbors1, k) = %d\n", response);
	assert(getResponse(neighbors1, k) == 0);

	// class vote of 2 same items is the same one class
	printf("== Test 2 ==\n");
	Iris neighbors2[2];
	k = 2;
	neighbors2[0] = new1;
	neighbors2[1] = base1;
	assert(getResponse(neighbors2, k) == 0);

	printf("== Test 3 ==\n");
	Iris neighbors3[2];
	k = 2;
	neighbors3[0] = new2;
	neighbors3[1] = base2;
	assert(getResponse(neighbors3, k) == 1);

	// class vote of 3 items, 2 diff classes, take the majority
	printf("== Test 4 ==\n");
	Iris neighbors4[3];
	k = 3;
	neighbors4[0] = new1;
	neighbors4[1] = new3;
	neighbors4[2] = base1;
	assert(getResponse(neighbors4, k) == 0);

	printf("All tests passed!\n");
	printf("You are awesome!\n");
}

// Simple selection sort, courtesy of Jas (slightly modified)
// lo..hi sort on int arrays
/*
void selectionSortInt(int a[], int lo, int hi){
   int i, j, min;
   int tmp;

   for (i = lo; i < hi; i++) {
      min = i;
      for (j = i+1; j <= hi; j++) {
         if (a[j] < a[min]) min = j;
         //if (less(a[j],a[min])) min = j;
      }
      //swap(a[i], a[min]);
      tmp = a[i];
      a[i] = a[min];
      a[min] = tmp;
   }
}
*/

// Calculate a predicted response based on neighbors
// Get each neighbor to "vote" for a class, the class with the most number of votes is the response
int getResponse(Iris* neighbors, int k) {
	int *classVotes = calloc(NUM_CLASSES, sizeof(int));	// initialize sums to 0
	assert(classVotes != NULL);
	int mostVotes;	// class w/ the most # of votes

	int i;
	int response;	// response per neighbor

	// get the class votes of each neighbor
	for (i = 0; i < k; i++) {
		response = neighbors[i].class;
		classVotes[response] += 1;
	}

	// sort the classVotes from lo.. hi
	// don't sort the array, the array indices are the class you want to return
	//selectionSortInt(classVotes, 0, k-1);

	// find the index with the most votes
	mostVotes = 0;
	for (i = 0; i < NUM_CLASSES; i++) {
		if (classVotes[i] > classVotes[mostVotes]) {
			// this class has more votes
			mostVotes = i;
		}
	}

	// take the highest classvote
	//mostVotes = classVotes[k-1];
	free(classVotes);
	return mostVotes;
}

// Very simple testing function for getNeighbors() function
void testGetNeighbors (void) {
	printf("Testing getNeighbors()...\n");
	Iris new1 = {5.1, 3.5, 1.4, 0.2, 0};
	//Iris new2 = {7.0,3.2,4.7,1.4,1};
	Iris base1 = {1, 2, 3, 4, 0};
	Iris base2 = {1, 2, 3, 4, 1};

	Iris *neighbors;	// for return output
	int k;	// for chosen k
	int trainSize;	// for training arrays

	// 1 nearest neighbor of 1 thing is itself
	printf("== Test 1 ==\n");
	Iris *train1 = {&base1};
	trainSize = 1;
	k = 1;
	neighbors = getNeighbors(train1, trainSize, base1, k);
	printf("Input array: \n");
	showIrisArray(train1, trainSize);
	printf("Output neighbors:\n");
	showIrisArray(neighbors, k);

	// 2 nearest neighbors of 2 things are the 2 things
	printf("== Test 2 ==\n");
	Iris train2[2];
	train2[0] = base1;
	train2[1] = base2;
	trainSize = 2;
	k = 2;
	neighbors = getNeighbors(train2, trainSize, base2, k);
	printf("Input array: \n");
	showIrisArray(train2, trainSize);
	printf("Output neighbors:\n");
	showIrisArray(neighbors, k);

	// 2 nearest neighbors of 3 things
	printf("== Test 3 ==\n");
	Iris train3[3];
	train3[0] = base1;
	train3[1] = new1;
	train3[2] = base2;
	trainSize = 3;
	k = 2;
	neighbors = getNeighbors(train3, trainSize, base1, k);
	printf("Input array: \n");
	showIrisArray(train3, trainSize);
	printf("Output neighbors:\n");
	showIrisArray(neighbors, k);

	free(neighbors);

	printf("Testing finished. Do visual checks.\n");
}

// Simple selection sort, courtesy of Jas (slightly modified)
// lo..hi sort on IrisSort arrays
void selectionSort(IrisSort a[], int lo, int hi){
   int i, j, min;
   IrisSort tmp;

   for (i = lo; i < hi; i++) {
      min = i;
      for (j = i+1; j <= hi; j++) {
         if (a[j].dist < a[min].dist) min = j;
         //if (less(a[j],a[min])) min = j;
      }
      //swap(a[i], a[min]);
      tmp = a[i];
      a[i] = a[min];
      a[min] = tmp;
   }
}

// Calculate the k most similar instances (neighbors) and return it in an array
Iris *getNeighbors (Iris* training, int trainSize, Iris testInstance, int k) {
	Iris *neighbors = malloc(k * sizeof(Iris));
	assert(neighbors != NULL);

	// calculate the k most similar instances to the given testInstance
	// out of the training array

	// keys are the distances of each instance to testInstance (exclude same instance?)

	// struct with pointer to training Iris and the key dist

	// array of IrisSort structs so we can calculate distances and sort them
	IrisSort *trainSort = malloc(trainSize * sizeof(IrisSort));
	assert (trainSort != NULL);

	// initialize all the pointers to Iris items
	int i;
	for (i = 0; i < trainSize; i++) {
		trainSort[i].item = &training[i];

	}

	// calculate distance of each Iris item to the testInstance
	for (i = 0; i < trainSize; i++) {
		trainSort[i].dist = euclidDist(*(trainSort[i].item), testInstance);
	}

	// sort (lo .. hi) trainSort array based on dist key
	selectionSort(trainSort, 0, trainSize - 1);

	// select the first k instances and copy into neighbors array to return
	for (i = 0; i < k; i++) {
		neighbors[i] = *(trainSort[i].item);
	}

	free(trainSort);

	return neighbors;
}

// testing function for Euclidean distance
void testEuclidDist (void) {
	Iris new1 = {5.1, 3.5, 1.4, 0.2, 0};
	Iris new2 = {7.0,3.2,4.7,1.4,1};
	Iris base1 = {1, 2, 3, 4, 0};
	Iris base2 = {1, 2, 3, 4, 1};

	assert(euclidDist(base1, base2) == 0);
	printf("euclidDist(new1, new2) == %lf\n", euclidDist(new1, new2));
	//assert(euclidDist(new1, new2) == 16.03);	// will fail due to float representation
}

// Euclidian distance function to calculate similarity between data points
// Only take into account first 4 fields of struct (not the class)
double euclidDist (Iris data1, Iris data2) {
	double distance;
	// break into lines just for readability
	distance = pow((data1.sepLen - data2.sepLen), 2);
	distance += pow((data1.sepWidth - data2.sepWidth), 2);
	distance += pow((data1.petLen - data2.petLen), 2);
	distance += pow((data1.petWidth - data2.petWidth), 2);
	return distance;
}

// Testing function for loadData()
// Prints to screen and do a visual check
void testLoadData (Iris *training, Iris *test, int trainSize, int testSize) {
	printf("Printing training array...\n");
	showIrisArray(training, trainSize);
	printf("Printing test array...\n");
	showIrisArray(test, testSize);
}

// loads the data in the file pointer into the training and test arrays, splitting it
// by the given split ratio 
void loadData (FILE *f, int split, Iris *training, Iris *test, int* trainIx, int* testIx) {
	// malloc cuz will pass into a function
	char* line = malloc(MAX_CHARS * sizeof(char));	// holder string for each line of input
	assert (line != NULL);
	int decision;		// random number mod 100
	*trainIx = 0;
	*testIx = 0;

	// realloc checks, counter for number of times you've realloced
	int numTrain, numTest;
	numTrain = 0;
	numTest = 0;

	// seed the rand() function
	srand(time(NULL));	// bugs may arise with seeding problem
	
	// read until no more input in the file
	while (fgets(line, MAX_CHARS, f) != NULL) {
		
		// split data set randomly into train and test data sets
		decision = rand() % 101;	// 101 for percentage range 0..100
		if (decision <= split) {
			// put this line in the training set
			storeLine(line, training, *trainIx);
			(*trainIx)++;
		} else {
			// put this line in the test set
			storeLine(line, test, *testIx);
			(*testIx)++;
		}

		// implement reallocs for size
		checkAvailability(training, *trainIx, &numTrain);
		checkAvailability(test, *testIx, &numTest);

		// check for infinite reallocing? If so, use counters
	}

	free(line);
}

// stores a line of input (CSV format)
// into the given set of Iris arrays at the given index
void storeLine (char* line, Iris* set, int index) {
	int numFields = 0;	// number of fields you've initialized'
	char* field = malloc(MAX_FIELD * sizeof(char));	// substring of each field in the CSV file
	assert (field != NULL);

	int fieldSize;	// to trim newline character

	// get first field item
	field = strtok(line, ",");
	while ((field != NULL) && (numFields < NUM_FIELDS)) {

		switch (numFields) {
		case 0:
			set[index].sepLen = atof(field);
			break;
		case 1:
			set[index].sepWidth = atof(field);
			break;
		case 2:
			set[index].petLen = atof(field);
			break;
		case 3:
			set[index].petWidth = atof(field);
			break;
		case 4:
			fieldSize = strlen(field);
			field[fieldSize - 1] = '\0';	// trim newline character

			//printf("field = %s\n", field);
			//printf("strcmp(field, SETOSA) = %d\n", strcmp(field, SETOSA));

			if (strcmp(field, SETOSA) == 0) {
				set[index].class = IRIS_SETOSA;
			} else if (strcmp(field, VERSICOLOR) == 0) {
				set[index].class = IRIS_VERSICOLOR;
			} else if (strcmp(field, VIRGINICA) == 0) {
				set[index].class = IRIS_VIRGINICA;
			} else {
				// check the data
				fprintf(stderr, "Flower class not one of the 3 given ones. Check data.\n");
				exit(0);
			}
			break;
		default:
			// shouldn't come here
			fprintf(stderr, "numFields = %d, should be less than %d\n", numFields, NUM_FIELDS);
			exit(0);
			break;
		}

		// get next field item
		field = strtok(NULL, ",");
		numFields++;
	}

	free(field);
}

// checks if the Iris array is full, reallocs INCREASE more spaces if it is
// the point in the set written to is given by index
void checkAvailability (Iris* set, int index, int *numRealloc) {
	
	int setSize = MAX_DATA + ((*numRealloc) * INCREASE);

	if (index >= setSize) {
		printf("I want to be realloced for some reason\n");
		set = realloc(set, (setSize + INCREASE) * sizeof(Iris));
		assert(set != NULL);
		(*numRealloc)++;
	}

}

// shows an Iris array pointed to by a with n items
void showIrisArray (Iris *a, int n) {
	int i;

	for (i = 0; i < n; i++) {
		// print each Iris struct as a tab separated line
		printf("%lf\t%lf\t%lf\t%lf\t%d\n", a[i].sepLen, a[i].sepWidth, a[i].petLen, a[i].petWidth, a[i].class);
	}
}

