# Programming-ML
This Folder has important use cases and projects that I have applied in the real word and also has important reference 
Here are the detailed notes that elobrate more 

- Data Analytics overview

    Role of data Scientist 

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d86889f7-1676-4931-ba96-024aa1ff68c5/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d86889f7-1676-4931-ba96-024aa1ff68c5/Untitled.png)

    Visualizing data and communicating with stakeholders is the most important steps as the data makes more progress, reasons being stakeholders will change the questions or will give an idea of what to ask the Scientist 

    ### Example of Data Science to provide relevant search recommendations

    1.Query Volume –Unique and verifiable users
    2.Geographical locations
    3.Keyword/phrase matches on the web
    4.Some scrubbing for inappropriate content
    **Example** : *E-Bay has literally stepped out from google for their ads bidding due to the impact of customers thinking the ADS will cost them money*

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ea5d4e96-70da-4062-ae76-e895db56a272/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ea5d4e96-70da-4062-ae76-e895db56a272/Untitled.png)

    **Data Wrangling** : This phase includes data cleansing, data manipulation, data aggregation, data split, and reshaping of data;
    •Erroneous data and Unexpected data format, outliers etc
    •Classifying data into linear or clustered
    •Determining relationship between observation, feature, and response without **personnel bias** 

    Data wrangling is the most challenging phase and takes up 70% of the Data Scientist’s time.

     **Data Exploration** : Data discovery and Data pattern

    •Based on the overall data analysis process, should be accurate to avoid iterations
    •Depends on pattern identification and algorithms
    •Depends on hypothesis building and testing
    •Leads to building mathematical statistical functions

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/205b7652-f515-42a8-a8b0-a8cf0d233870/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/205b7652-f515-42a8-a8b0-a8cf0d233870/Untitled.png)

    pyenv , pip and conda - the prominent packages in python and always everr goriwn 
    Annaconda to get the libraries faster - get togeather 

    ### **EDA**

    1. Number of unique values in each column?
    2. Value counts of all categorical columns.
    3. Univariate and Bivariate Analysis
    4. Get data ready for model development
- Basics of python [ list Comprehension , range function , lambda etc

    Make a habit of using range **(range (1,20,2))** for courting the numbers with intervals

    - **Change location/directory -** Usage to change if necessary before working

        ```python
        import os
        os.getcwd()

        **#we use this to get the current directory location and then we map as per our convenience**

        # WINDOWS
        os.chdir('C:\\Users\\Notebooks\\DSwithPython')
         Test below in its own cell: pwd

        #**change directory location** 
        ```

    - Any data that you load/initialize using Pandas will be represented in the form of DataFrame

        ```python
        Data.columns = ['E_id','Emp_Name','Emp_Sal']

        #You can replace Column indexes with column names, namming is right to left

        data = pd.read_csv("employee.csv",header=None)

        #We will get the colums name as numbers

        url = "http://www.basketball-reference.com/leagues/NBA_2015_totals.html"
        BB_data = pd.read_html(url)

        #WUsed to get the data from website

        ```

        - '**with**' statement -  Best practice use for auto closes the file for you

        ```r
        #  There are multiple ways to open a file.
        #  You can open it just for reading, or for writing or both
        #  When you open a file for writing it does one of two things
        #     If the file name already exists it overwrites the file 
        #        (meaning: it erases the contents so your file is now open but empty)
        #     If the file name does not already exist it creates a new empty file
        #  Place your cursor next to the 'open' command and press your shift+tab to see mode values
        #  hit shift-tab to get information about command
        with open ('myfile2.txt') as myfile2:
            contents = myfile2.read()
        contents
        ```

        ### Lambda function : it will be a short version recurring function so that we can quickly induce in apply function. we dont have to define and return the result explicitly.

        ```python
        #used for recursive calling 
        Demo = lambda n: n* Demo(n-1) if n!=0 else 1 

        **#ConventionalWay** 

        def myAdd(n1,n2):
            result=n1+n2 
            return result 

        **#Variables and code in the returnStatement!!**

        myAdd_v2 = lambda n1,n2: n1+n2

        Demo = lambda n: n* Demo(n-1) if n!=0 else 1 

        ```

    - **Back to the shell game -** Best way to learn interaction with functions

    ### List Comprehension

    # 1. [ *Return_Expression* for item in list if (condition) ]
    # 2. [ *Return_Expression1* if (condition) else (condition) *Return_Expression1* for item in list]

    ```python
    l=list(range(1,20,1))

    results = [i for i in l if i%2==0]
    results

    **#the return statement is written towards the left side**
    ```

- Numpy

    Generally we use Numpy to use stat functions like Np.Sum, Count, Max, Mode, argMin, zero, one, slicing, sub setting, reshaping , flatten and linespace 

    - Numpy Arrays - Wrapping around math concept via python way

        ```python
        # 2.1 Create 2 python lists 
        distance = [10,15,17,26]
        time = [.30,.47,.55,1.20]
        # 2.3 import numpy as np
        import numpy as np
        # 2.2 Create 2 ndarrays using np.array(NameOfList)
        np_distance = np.array(distance)
        np_time = np.array(time)
        Simple math on two ndarrays
        # 2.4 Test doing math on two ndarrys
        speed = np_distance / np_time
        speed
        ----------------------------------------------------------
        method: array COMMON MISTAKE
        # 2.5 A common mistake when creating an array
        #     Must use a sequence to create ndarray
        # a = np.array(1,2,3,4)    # WRONG
        a = np.array([1,2,3,4])  # RIGHT

        ```

    - Images are read in terms of array for the system perception, Numpy is used for the summary stats like shape, reshaping & resizing resolution , super-imposing and other functions

    [Reshaping numpy arrays in Python - a step-by-step pictorial tutorial](https://towardsdatascience.com/reshaping-numpy-arrays-in-python-a-step-by-step-pictorial-tutorial-aed5f471cf0b)

    - We can create Matrix(np.matrix()) and Array(np.array()) using Numpy, big difference between *Matrix and Array*, Matrix is a 2 dim array and also can be N dim and works on dot product

    ```python
    np.array(list1)

    array1.argmax() #index location of the max value

    array1.max() #max value in the array
     
    randint( startrange, endrange, number of elements) #Returns only integer values wiithin the range specified
    ```

- Pandas

    we pass the last through the dataframe so that we can make changes to the dataframe with different functions

    Series is a column from the dataframe. Very similar to **Numpy Array with dimension (X,1)**

    ```python

    ****employeeDataFrame = pd.read_csv("C:/Users/Rahul Aggarwal/Desktop\\employee.csv",header=None)

    Example = pd.read_clipboard(sped='-') #**Read data from the clipboard**

    **#we pass our excel sheet as dataframe and then work on it further 

    #Pandas Dataframes are mutable in nature you can create new varables on the go and mutate exsisting var**

    empWithHeaderDF['NewSalary'] = yearlySalary

    #**Pandas Dataframes can be deleted and inplace is used for perminant**

    empWithHeaderDF.drop(['NewSalary, 'ename'],axis=1,inplace=True)
    ```

    - Converting DF into Numpy Array
    - **DATA TRAVERSING using iloc and other logical operations**

    Pandas recommends to create a **Function that can implement the logic** and use apply method to implement function in a series. Reason being we will not be able to apply the series through a function at once, as function takes only one value at a time 

    **Data Subsetting** : in order to subject the data we use the , iloc --> Will get index Location  SYNTAX ,--> iloc[rowIndex/rowRange/rowList , columnIndex/columnRange/columnList ]

    ```python
    empWithHeaderDF['UpdatedYearlySalary'] = empWithHeaderDF['esal'].apply(incrementSalary)

    #**Pandas Dataframes series cant be function hence we use apply function!!!** 

    empWithHeaderDF.to_csv('C:\\Users\\Rahul Aggarwal\\Desktop\\OutputFileJan10_New.csv',index=None)

    ```

- Functions and codes used frequently

    [Functions used frequently ](https://www.notion.so/1a18aef8fe8c4381a242c277bb888411)

- Data visualization
    - **Matplotlib** is the core visualization library in Python. **Seaborn** uses Matplotlib as base library [ *ggplot** is the core library from R language, it gives similar functionality in Python ] import matplotlib.pyplot as plt*

    - **Relationship between histogram and barplot** : Histogram will convert the continuous variable into buckets and then computes frequency .
    - linspace is an in-built function in Python's NumPy library. It is used to create an evenly spaced sequence in a specified interval. *( used in the histogram behind the scenes )*

    **Correlation quantifies the direction and strength of the relationship** between two numeric variables, X and Y, and always lies between -1.0 and 1.0. Simple linear regression relates X to Y through an equation of the form Y = a + bX. When the correlation (r) is negative, the regression slope (b) will be negative.

    - Data visualization program

    [28 Jupyter Notebook Tips, Tricks, and Shortcuts for Data Science](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)

    - Matlab
- Linear Regression ( need to make notes more better on lasso etc)

    > Features and Lables must be in Numpy array and also must be in 2D format - we can use the Repshape method to get the data set in the 2D format

    Concept of Regularization regression - need to work on this further

    - Single Linear Regression
    - Multiple Linear Regression along with Regularization concept
- Polynomial regression - non Linear regression model

    Pre processing steps are applicable for the features of the Data set for both X_train and X_test and encoding is applicable for the Target values. Never pre process data of NAN which are in Target

    ```python
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    Data = pd.read_csv('data (1).csv')
    Data.info()
    Data.describe()
    Data
    ---
    print("Null values:", Data['Pressure'].isnull().sum())
    zscore = (Data.Pressure - Data.Pressure.mean()) / Data.Pressure.std()
    #If the z score > 3, Such a data point can be an outlier
    outliers = zscore.loc[abs(zscore) > 3]
    print("Outlier count:", len(outliers))
    ----
    features=Data.Temperature.values #or - features= Data.iloc[:,0].values
    labels=Data.Pressure.values #or -labels=Data.Pressure.values
    features.shape
    labels.shape

    #better to use the reshape for regression
    labels=labels.reshape(6,1)
    features=features.reshape(6,1)
    labels.shape
    features.shape
    plt.scatter(features, labels)

    #What if we apply the linear regression to this problem set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(features,labels)
    regressor.score(features,labels)

    #from the plot we can check that it will not follow any proper prediction pattern - UNDERFITTING
    plt.scatter(features,labels,color='blue')
    plt.plot(features,regressor.predict(features),color='red')

    #This method will raise the polynomial degree so that we can work on the transfomred data

    from sklearn.preprocessing import PolynomialFeatures
    #root of the equation would be the degree, we need to find the sweet spot by checking the score
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(features)
    regressor.fit(x_poly,labels)

    #line now with the use of rasied features
    plt.scatter(features,labels)
    plt.plot(features,regressor.predict(x_poly), color='red')
    regressor.score(x_poly,features)

    ```

- Logistics Regression  - Binary Classification Model

    ### Precision , Recall , F-score and Accuracy :  *( Remember the Dart example )*

    **Precision** : If the Precision is larger then its the spread of value will be around , but it doesn't mean its accurate as it could be anywhere 
    **Accuracy** :  How correctly the labels have been specified for the model 
    **Recall** : sensitivity of the model, how well the model is able to predict 

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1e99a2c-610d-4080-ba91-0400fffd5c52/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c1e99a2c-610d-4080-ba91-0400fffd5c52/Untitled.png)

    ## Rules for Classification

    1. Data must be complete with no null values
    2. Features must be strictly numerical. Labels can be numeric or non-numeric
    3. For Logistics Regression, the label must be binary 
    4. Data should be in Numpy array and features should be in 2 D array, Lables will be 1 D array 
    - Example :  Social_Network_Ads
- Decision Tree  - Multiclass Classification Model

    ## **Decision Tree Algorithm Pseudocode**

    - Place the best attribute of the dataset at the root of the tree. (
    - Split the training set into subsets. Subsets should be made in such a way that each subset contains data with the same value for an attribute.
    - Repeat step 1 and step 2 on each subset until you find leaf nodes in all the branches of the tree

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2a924869-5634-4c7a-8ed8-3ea489c17390/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2a924869-5634-4c7a-8ed8-3ea489c17390/Untitled.png)

    **IMPURTIY** : Measure of *disorder, or uncertainty*
    **ENTROPY** : Entropy controls how a Decision Tree decides to split the data. It actually effects how a Decision Tree draws its boundaries. The splitting work through the concept of **Entropy "***is the measures of impurity, disorder, or uncertainty in a bunch of examples*"
    **INFORMATION GAIN** :  how much information goes for the outlook , Information gain is used to decide which feature to split on at each step in building the tree. A commonly used measure of purity is called information gain. After these factors are found out, the *root node is split based on the purest homogeneous split  ( based on the values of Information Gain)* 

    If we have a set of candidate covariates **from which to choose as a node in a decision tree,** we should choose *the one that gives us the most information about the response variable (i.e. the one with the highest entropy).

    For the case of regression, **the Splitting factor would be based on the variance,** the model which will give us the **lowest variance  and MSE value** will get us the best feature for homogenous splitting* 

    - Decision Tree sample example
- Ensemble

     It is a technique used for improving prediction accuracy. It uses multiple learning algorithms instead of a single algorithm.

    Characteristics of a good model:
    1) Bias - Lower
    2) Variance - Lower

    ### Common Ensemble Techniques:

    - Bagging :  Bootstrapping + Aggregating

    **Bootstrapping :** is a sampling method, where a sample is chosen out of a set, using the replacement method. The learning algorithm is then run on the samples selected. The bootstrapping technique uses sampling with replacements to make the selection procedure completely random.

    **Aggregation :** Model predictions undergo aggregation to combine them for the final
    prediction to consider all the outcomes possible. The aggregation can be done based on the total number of outcomes or on the probability of predictions derived from the bootstrapping of every model in the procedure.

    The ensemble model based in bagging reduces the variance of the estimate (target column prediction value).

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/858c08ac-3831-4006-ba24-7d8998c57711/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/858c08ac-3831-4006-ba24-7d8998c57711/Untitled.png)

    - Bagging Program using  IRIS DATASET

    **Boosting** :

    Boosting considers homogenous weak learners and it learns from them in a sequential manner. Unlike bagging, where the learning is parallel.  Here, the learning is also adaptive -- the base model depends on the previous model's output.

    ### ADABoosting : Adaboost identifies the shortcomings by adding a high weight for the misclassified data points.

    1. **First we assign the common weight** : 1/samplesize
    2. **Formula to get significance**  :  1/2 log(1-totalerror/total error) 
    3.  **New sample weight** : sample weight * e^significance ( we classify for the misclassified as e^-significance )
    4. The **new weights** obtained will not be equal to 1, hence we **normalize** it via standardscalar or minmax Scalar 
    5. We will go on with next feature and project the same on the Target and calculate new weight and normalize it and this iteration will go on ( based on the iteration given)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f66b4e17-1f95-48bc-92d3-8875f1575629/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f66b4e17-1f95-48bc-92d3-8875f1575629/Untitled.png)

    ### XGboost through the concept the of Gradient descent  :  it uses a gradient in the loss function to identify the misclassified points.

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c7fb207-14ee-4cbb-93b2-2b987ff8df44/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c7fb207-14ee-4cbb-93b2-2b987ff8df44/Untitled.png)

    ### **Cross-Validation : It used on limited sample and the predictions on data, More efficient use of data(every observation is used for both training and testing)**

    **Characteristics of Model:**
    Low Bias
    Low Variance

    **Characteristics of Train Data:**
    Low Bias
    High Variance

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3e2f508-8d14-4533-8b65-eb0eef65abeb/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a3e2f508-8d14-4533-8b65-eb0eef65abeb/Untitled.png)

    - **Boosting program : ADAboosting, XGboosting and Cross Validation**
- Random Forest

    Random forest will have the different instances of the Decision Tree, where each of it will be **contributing to its own independent entropy division**. Its very good when we have many **missing values**.  **( Better if its used for Ensemble )**

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5e64c37-e830-4048-af49-75676b8d24d7/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c5e64c37-e830-4048-af49-75676b8d24d7/Untitled.png)

    The resulting dataset from Iris is a Bunch object, (A Bunch object is similar to a dicitionary, but it additionally allows accessing the keys in an attribute style) hence we by using `iris.keys()`we get the below Output, we will be able to use either of them to retrieve the categorical values 

    `dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])`

    > Iris data is from of bunch object hence we will have to use a different approach

    - Example : Random forest using Iris dataset

        ```python
        #IRIS` Use case

        import pandas as pd
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        %matlplotlib in line 
        import sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import RandomForestClassifier  
        np.random.seed(0)

        iris = load_iris()
        #The date of iris is in the form of the Dict hence we will have to put them in dataframe with VAR
        df=pd.dataframe(iris.data, colums=iris.features_names)
        df.head
        #getting the answer for these exsiisiting ones to train the model
        df['species']= pd.Categorical.from_codes(iris.target)
        df.head()
        #random number between 0 and 1 for each row ad we are splitting the data into train and test cases with 75%
        df['is_train'] = np.random.uniform(0,1,len(df) <- .75

        #reating the dataframe with test row and training rows
        train, test = df[df['is_train']==True],df[df['is_train']==False]
        len(train)
        len(test)

        #all the indep var which form the training test
        features = df.colums[:4]
        print(features)

        #here the catagory data is converted to computer readable format
        y =pd.factorize(train['species'])[0]
        y

        # n_jobs will set the priority for the code
        clf=RandomForestClassifier(n_jobs=2, random_state=0) 
        #feeding the model with the train test and already known predicitions
        clf.fit(train[features],y)
        #inserting the test(25%) that we have split earlier and features would be the indep VAR col
        clf.predict(test[features])
        #we are checking the predict in numberical value as it would be used for assigning the priority for random forest
        clf.predict_proba(test[features])

        #here we are setting the names of the result to the catagorical var and this is model prediction
        preds = iris.target_names[clf.predict(test)]
        preds[0:5]
        #while this would be for the base model  results
        test.species.head()
        #confusion Matrix
        pd.crosstab(test['species'], preds,rownames=['Actual Species'], colnames=['perdicted Species'])
        #we can send values in the 
        preds = iris.target_names[clf.predict([5.0,3.6,1.4,2.0])]
        ```

- Naive Bayes classification  - Multiclass Classification Model

    **Bayes theorem ( mulitclass problems )**: Where: P(A|B) – the probability of event A occurring, given event B has occurred. P(B|A) – the probability of event B occurring, given event A has occurred. P(A) – the probability of event A.

    #too many categorical values will impart the results of the  Naive Bayes

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c92e62ed-2f4c-4953-b08c-b1057f9f3405/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c92e62ed-2f4c-4953-b08c-b1057f9f3405/Untitled.png)

    - Example : Naive Bayes classification using Iris
- Support vector machine  - Multiclass Classification Model

    Support vector machine is very good for multiple categorical features 

    Multiple lines of best fit are drawn to classify the model ( Hyper-plane in 3D space ) , it is calculates on the basis of maximum separation via support vectors ( extremities of on the features given)

    A Kernel Trick is a simple method where a Non Linear data is projected onto a higher dimension space so as to make it easier to classify the data where it could be linearly divided by a plane

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eea2b3e3-4a18-481c-bf1f-64f7a28de3ef/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eea2b3e3-4a18-481c-bf1f-64f7a28de3ef/Untitled.png)

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a43df3a9-c804-4e34-9371-f6047613b0c6/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a43df3a9-c804-4e34-9371-f6047613b0c6/Untitled.png)

    - SVM - Project
- Hierarchical clustering - Unsupervised learning

    > Euclidean Distance and Manhattan Distance difference - Check the below video

    [Euclidean Distance and Manhattan Distance](https://www.youtube.com/watch?v=p3HbBlcXDTE)

    1) Along the distance matrix the diagonals would be 0.
    2) Euclidean Distance would be found out for all the values. (Values on the either side of the diagonal would be similar.
    3) The min distance would be merged, in our case as per example is CD, and again the Euclidean distance would be found out for the new values formed 
    4) Using the distance matrix and we will be able to use either ***Mean linkage, Max or Complete Linkage , Centroid Linkage , Min or Single Linkage** to compute the distance after the Merge.* 
    5) Considering the below example the of CD merge, ( if we take the method of ***Max or Complete Linkage***  we can see that the among C and D, we take C as it has max distance value
    6) The iteration goes on and so forth

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/884c35b8-8df6-4e27-b55d-c0ecb17dbe29/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/884c35b8-8df6-4e27-b55d-c0ecb17dbe29/Untitled.png)

    - Hierarchical clustering - Zoo example
- K-means clustering - Unsupervised learning

    1)  Initializing the centers Randomly from the data points
    2) Assignment of the points to the nearest center that are taken randomly  
    ****3) We take an equidistance from all the points i.e Centroid ( *we take centroid for irregular ****geometry figures* ) 
    4) Assignment of the new points as per the new center -  **this process keeps iterating till there is no readjustment** 
    5) For the assignment of the clusters take the lowest possible - we arrive at this through the elbow method

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b221ded8-05c3-408d-bea7-b9c6872218f4/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b221ded8-05c3-408d-bea7-b9c6872218f4/Untitled.png)

    - Elbow and Silhouette methods are used to find the optimal number of clusters. Ambiguity arises for the elbow method to pick the value of k. Silhouette analysis can be used to study the separation distance between the resulting clusters and can be considered a better method compared to the Elbow method.
    - Elbow method will not be able to Gurage the overlapping and also will not be able to find the intra cluster distance and also the neighboring cluster

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74d660cc-2144-4b45-a054-7a84ee825c6f/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/74d660cc-2144-4b45-a054-7a84ee825c6f/Untitled.png)

    - K-Means Clustering - Drive Data Algorithm

- Feature Engineering
    1. Wait for the model to build (Trade-off Training Time)
    2. Reduce the number of variables by merging correlated variables.
    3. Extract the most important features that are responsible for maximum variance in the output.

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eff76c98-1fb9-4274-a20a-3a7b701ea60e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/eff76c98-1fb9-4274-a20a-3a7b701ea60e/Untitled.png)

    ### Factor Analysis:

    Feature Variance - PCA , while Correlation between Feature and Target - LDA

    - **Uniformly distributed** - LDA performs (no outliers , skewness) better than PCA (*LDA - can be applied only on labelled*)
    - **Irregular Distribution** - Use PCA it can take SkewNess well (*PCA - can be applied on both labelled and unlabelled*)
    - PCA Program on Iris Data Set
    - LDA Program on Iris Data Set
- Projects Insights

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/983f2bdd-d273-4605-a2a6-65d831b12a97/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/983f2bdd-d273-4605-a2a6-65d831b12a97/Untitled.png)

    In the below pic, based on the skewness and also based on values ( which are non 0 and 0 )  we apply log Transformation 

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f9024d3-36f4-4a7b-a4de-98b2b25f5f16/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8f9024d3-36f4-4a7b-a4de-98b2b25f5f16/Untitled.png)
