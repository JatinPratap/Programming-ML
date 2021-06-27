#!/usr/bin/env python
# coding: utf-8

# <hr>
# <h3>Axe Bank Credit Card Customer Segmentation</h3>
# <hr>
# 
# <b>Background</b>: Axe Bank wants to focus on its credit card customer base in the next financial year. They have been advised by their marketing research team, that the penetration in the market can be improved. Based on this input, the Marketing team proposes to run personalised campaigns to target new customers as well as upsell to existing customers. Another insight from the market research was that the customers perceive the support services of the back poorly. Based on this, the Operations team wants to upgrade the service delivery model, to ensure that customers queries are resolved faster. Head of Marketing and Head of Delivery both decide to reach out to the Data Science team for help.
# 
# <b>Data Description</b>: Data is of various customers of a bank with their credit limit, the total number of credit cards the customer has, and different channels through which customer has contacted the bank for any queries, different channels include visiting the bank, online and through a call centre. 
# 
# <b>Key Questions:</b> 
# 1. How many different segments of customers are there?
# 2. How are these segments different from each other?
# 3. What are your recommendations to the bank on how to better market to and service these customers?

# ### Importing the Libraries

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler

import seaborn as sns 
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage,cophenet


# In[3]:


#Reading the dataset 
df=pd.read_excel('CreditCardCustomerDataSet.xlsx')
df.shape


# In[4]:


df.describe()


# #### There are only 655 unique values in "customer Key", it is likely that there are few values that are missing 

# In[5]:


#number of credit cards owened credit cards so far
df['Total_Credit_Cards'].value_counts()


# In[6]:


df['Total_visits_bank'].value_counts()


# In[7]:


#pie chart for the target value

plt.figure(figsize= (12,7))
df_target= df['Total_visits_bank'].value_counts()
plt.pie(df_target, labels= df_target.index, autopct= '%.1f%%', startangle= 90, shadow = True )

plt.title('Total visits bank', fontsize= 16)

plt.show()


# In[8]:


df['Total_visits_online'].value_counts()


# In[9]:


#pie chart for the target value

plt.figure(figsize= (12,12))
df_target= df['Total_visits_online'].value_counts()
plt.pie(df_target, labels= df_target.index, autopct= '%.1f%%', shadow = True )

plt.title('Total visits online', fontsize= 16)

plt.show()


# In[10]:


df['Total_calls_made'].value_counts()


# In[11]:


#pie chart for the target value

plt.figure(figsize= (12,12))
df_target= df['Total_calls_made'].value_counts()
plt.pie(df_target, labels= df_target.index, autopct= '%.1f%%', shadow = True )

plt.title('Total calls made', fontsize= 16)

plt.show()


# In[12]:


pd.crosstab(df['Total_calls_made'], df['Total_visits_bank'],normalize='columns')


# In[13]:


pd.crosstab(df['Total_Credit_Cards'], df['Total_visits_bank'],normalize='columns')


# In[14]:


#number of unique customers with the bank 
df['Customer Key'].nunique()


# In[15]:


df['Customer Key'].value_counts()


# In[16]:


df[df['Customer Key']==50706]


# In[17]:


df['Customer Key'].drop_duplicates(keep='last').shape


# In[18]:


# Dropping duplicates based on unique customer key
df = df.iloc[df['Customer Key'].drop_duplicates(keep='last').index]
df.shape


# The cols : Sl_No and CustomerKey are IDs which can be eliminated as they are unique and will not have any relevant role in forming the clusters so we remove them

# In[19]:


cols_to_consider=['Avg_Credit_Limit','Total_Credit_Cards','Total_visits_bank','Total_visits_online','Total_calls_made']


# In[20]:


subset=df[cols_to_consider]  #Selecting only the above columns 


# In[21]:


subset


# ### EDA 

# #### Checking for Missing Values 

# In[22]:


#NullValues with the help of Heatmap

sns.heatmap(subset.isnull(), cmap="inferno")
plt.figure(figsize=(16,9))


# No missing values were found 

# #### Checking for the statistically summary 

# In[23]:


subset.describe()


# The min and max value of 'Avg_Credit_Limit' is very larger as compared to the other columns 
# To bring the data to the same scale let's standardize the data.
# 
# 

# ## Feature Correlations

# In[24]:


# Use Corr function to create correlation matrix
subset.corr()


# **Plot Correlation Matrix**

# In[25]:


## Use Seaborn Heatmap to visualize correlation 
sns.heatmap(subset.corr(),annot=True,vmin=-1,vmax=1);


# ## Visualize feature distributions

# In[26]:


sns.pairplot(subset,diag_kind='kde');


# In[27]:



## distribution among continuous values

for i in subset:
    data=subset.copy()
    data[i].hist()
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()


# ## Check Skewness

# In[28]:


subset.skew()


# ### Log Transformation [Box-Cox Transformation]

# In[29]:


subset_2=subset.copy()


# In[30]:


# Use Log transformation to scale features

## Hint : use np.log function 

subset_2['Avg_Credit_Limit'] = np.log(subset_2['Avg_Credit_Limit']) 

#can't take log(0) and so add a small number

subset_2['Total_visits_online'] = np.log(subset_2['Total_visits_online']+0.1)


# In[31]:


subset_2.skew()


# ## Visualize the Normalized data

# In[32]:


# Produce a scatter matrix for each pair of features in the data
sns.pairplot(subset_2,diag_kind='kde');


# In[33]:


sns.heatmap(subset_2.corr(),annot=True);


# In[34]:


subset_2


# ##  Feature Scaling For Standardization -  Standard Scaler ( Z Score )

# In[35]:


scaler=StandardScaler()
subset_scaled=scaler.fit_transform(subset_2)   


# In[36]:


subset_scaled_df=pd.DataFrame(subset_scaled,columns=subset_2.columns)   #Creating a dataframe of the above results


# In[37]:


subset_scaled_df


# In[38]:


subset_scaled_df.skew()


# ## Execute K-Means Algorithm

# In[39]:


## Iterate the K-Means for different values of clusters. Compute the error term and store in an object

cluster_range = range( 1, 15)
cluster_errors = []

for num_clusters in cluster_range:
    clusters = KMeans( num_clusters, n_init = 100,init='k-means++')
    clusters.fit(subset_scaled_df)
    cluster_errors.append( clusters.inertia_ )    # capture the intertia


# In[40]:


# combine the cluster_range and cluster_errors into a dataframe by combining them
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors} )
clusters_df


# ## Elbow Method

# In[41]:


plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" );


# ## Execute the K-Means again with optimal cluster number 

# In[42]:


kmeans = KMeans(n_clusters=3, n_init = 15, random_state=2345)
kmeans.fit(subset_scaled_df)


# In[43]:


centroids = kmeans.cluster_centers_ 
centroids


# In[44]:


centroid_df = pd.DataFrame(centroids, columns = subset_scaled_df.columns )


# In[45]:


centroid_df


# The above are the centroids for the different clusters 

# #### Adding Label to the dataset

# In[46]:


dataset=subset_scaled_df.copy()  #creating a copy of the data 


# In[47]:


dataset


# In[48]:


dataset['KmeansLabel']=kmeans.labels_


# In[49]:


dataset


# In[50]:


dataset.groupby('KmeansLabel').mean()


# ## Customer Profiling - Visualizing the clusters

# In[51]:


sns.pairplot(dataset,diag_kind='kde',hue='KmeansLabel');


# In[52]:


subset['KmeansLabel']=kmeans.labels_
subset


# In[53]:


subset.groupby('KmeansLabel').mean()


# ### The clusters we are visualizing seems to do a good job but the preferred way will be to reduce the dimensions to 3 or less (if possible )  and then try to plot the clusters
# HINT: Try PCA before clustering 

# ### Analyse the Clusters 

# Let us make a visualization to observe the different clusters by making boxplots , 
# for the clusters we expect to observe statistical properties which differentiates clusters with each other 

# In[54]:


dataset.boxplot(by = 'KmeansLabel',  layout=(2,4), figsize=(20, 15))
plt.show()


# Looking the box plot we can observe differentiated clusters 

# ## Silhoutte Analysis For K-Means Clustering

# In[55]:


from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)


# In[56]:


dataset


# In[57]:


X=dataset.drop('KmeansLabel',axis=1).values
y=dataset['KmeansLabel'].values

range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters,n_init = 100,init='k-means++',random_state=0)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values =             sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.Spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors)

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1],
                marker='o', c="white", alpha=1, s=200)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()


# # <center> Hierarchical Clustering 

# Now that we have tried Kmeans , let's try hierarchical clustering with different dendograms for the same dataset and choosing the best using the cophenetic coefficient by using different types of linkages

# In[58]:


linkage_methods=['single','complete','average','ward','median']
results_cophenetic_coef=[]
for i in linkage_methods :
    plt.figure(figsize=(15, 13))
    plt.xlabel('sample index')
    plt.ylabel('Distance')
    Z = linkage(subset_scaled_df, i)
    cc,cophn_dist=cophenet(Z,pdist(subset_scaled_df))
    dendrogram(Z,leaf_rotation=90.0,p=5,leaf_font_size=10,truncate_mode='level')
    plt.tight_layout()
    plt.title("Linkage Type: "+ i +" having cophenetic coefficient : "+str(round(cc,3)) )
    plt.show()
    results_cophenetic_coef.append((i,cc))
    print (i,cc)


# In[59]:


results_cophenetic_coef_df=pd.DataFrame(results_cophenetic_coef,columns=['LinkageMethod','CopheneticCoefficient'])
results_cophenetic_coef_df


# Looking at the best cophenetic coefficient we get is for "Average" linkage.
# 
# But looking at dendogram 'ward' and 'complete' show good difference between clusters.
# 
# So choosing 'complete' because it has high cophenetic coefficirnt and good cluster segregation.
# 
# Lets make a dendogram for the last 25 formed clusters using complete linkage to have a better view since the above dendograms are very populated 

# In[60]:


#use truncate_mode='lastp' to select last p formed clusters
plt.figure(figsize=(10,8))
Z = linkage(subset_scaled_df, 'average', metric='euclidean')

dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=25 # show only the last p merged clusters
)
plt.show()


# Let's take a maximum distance around 5 to form the different clusters as clearly visible it cuts the tallest vertical lines.

# In[61]:


max_d=3.2
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, max_d, criterion='distance')


# In[62]:


set(clusters)  # So there are 3 clusters which are formed 


# ### Assign the clusters label to the  data set

# In[63]:


dataset2=subset_scaled_df[:] #Create a duplicate of the dataset


# In[64]:


dataset2['HierarchicalClusteringLabel']=clusters


# In[65]:


dataset2


# In[66]:


dataset2.groupby('HierarchicalClusteringLabel').mean()


# ### Analyse the clusters 

# In[67]:


dataset2.boxplot(by = 'HierarchicalClusteringLabel',  layout=(2,4), figsize=(20, 15))
plt.show()


# Here also we observe differentiated clusters.

# ### Silhouette Score

# In[68]:


from sklearn.metrics import silhouette_score
silhouette_score(dataset.drop('KmeansLabel',axis=1),dataset['KmeansLabel'])


# In[69]:


from sklearn.metrics import silhouette_score
silhouette_score(dataset2.drop('HierarchicalClusteringLabel',axis=1),dataset2['HierarchicalClusteringLabel'])


# Silhouette Score is better when closer 1 and worse when closer to -1
# 
# Here Kmeans score is slightly better tha Hierarchical

# ### Comparing Kmeans and Hierarchical Results

# In[70]:


Kmeans_results=dataset.groupby('KmeansLabel').mean()
Kmeans_results


# In[71]:


dataset.groupby('KmeansLabel').count()


# In[72]:


Hierarchical_results=dataset2.groupby('HierarchicalClusteringLabel').mean()
Hierarchical_results


# In[73]:


dataset2.groupby('HierarchicalClusteringLabel').count()


# #### Carefully observing the above results we can say that : 
# 

# 
# Cluster 0 of Kmeans appears similar to Cluster 2 of Hierarchical 
# 
# 
# Cluster 1 of Kmeans appears similar to Cluster 3 of Hierarchical 
# 
# 
# Cluster 2 of Kmeans appears similar to Cluster 1 of Hierarchical 
# 
# 

# #### Let's rename 

# 
# Cluster 0 of Kmeans  and Cluster 2 of Hierarchical as G1
# 
# Cluster 1 of Kmeans  and Cluster 3 of Hierarchical as G2
# 
# Cluster 2 of Kmeans  and Cluster 1 of Hierarchical as G3
# 
# 

# In[74]:


Kmeans_results.index=['G1','G2','G3']
Kmeans_results


# In[75]:


Hierarchical_results.index=['G3','G1','G2']
Hierarchical_results.sort_index(inplace=True)
Hierarchical_results


# In[76]:


Kmeans_results.plot.bar();


# In[77]:


Hierarchical_results.plot.bar();


# #### By both the methods of Clustering we get comparable clusters

# ## Cluster Profiles and Marketing Recommendation

# Since both the clustering alogrithms are giving similar clusters so we can assign labels from any one of the algorithm to the original (non scaled) data  to analyse clusters profiles
# ( here we are assigning labels of Kmeans , same could be done using hierarchical labels) 

# In[78]:


subset['KmeansLabel']=dataset['KmeansLabel']


# In[79]:


subset


# In[80]:


subset['KmeansLabel'].value_counts()


# In[81]:


subset.groupby('KmeansLabel').mean()


# #### Understanding each feature characterstics within different clusters 

# In[82]:


for each in cols_to_consider:
    print (each)
    print ( subset.groupby('KmeansLabel').describe().round()[each][['count','mean','min','max']])
    
    print ("\n\n")
    
    


# ### Analysis of clusters and questions answered :
#     

# #### 1. How many different segments of customers are there? 
# 
# Answer : Total numbers of segments are 3
#     
#     
#   

# #### 2. How are these segments different from each other? (Cluster profiles )
#   
# 
# Answer: 
#     
# 
# **Label 0 can be considered low valued customers**
#    
#     This group comprises of about 34% of the customers ( 224/660 )
#     
#     These customers have a mean "Avg_Credit_Limit " around 12200 and have 2 credit card on an average and the maximum number of credit card as 4.
#     
#     They are the ones who makes the most number of customer care calls to the bank as the average calls made is 7 
# 
# 
# 
# **Label 1 can be considered medium valued customers** 
#     
#     This group forms the majority of the customers having about 58% customers in total  ( 386/660 )
#     
#     These customers have  "Avg_Credit_Limit " ranging from 5000.0 to 100000.0 
#     
#     These are the ones which make the maximum number of visits to the bank as the average visits to bank is 3.
#     
#     They are the ones who are least active online as the maximum visit onine is just 3
# 
# 
# 
# **Label 2 can be considered  high value customers** 
#     
#     These are the least in number i.e. only 50 customers comprising 7.5% of total customers (50/660) .
#     
#     These customers have a minimum "Avg_Credit_Limit " of 84000 and have atleast 5 Credit cards .
#     
#     These are the ones which make the minimum number of visits to the bank as the maximum visit to bank is 1 amongst all 50     customers.
#     
#     They are mostly using online services as the average visit online is 11. 
# 
# 
# 

# #### 3. What are your recommendations to the bank on how to better market to and service these customers? (Business Recommendations )

# 1. Customers in the medium group ( having Label 1 ) are not engaged much in online activities , one of the exercise can be to engage them online. If they join online , promotions and offers can be communicated to them with much ease.
# 
# 
# 
# 2. Customers in low group ( label 0 ) can further be binned to check if there are any extreme groups having high average credit limit.These customers can be given more offers and new credit cards so that we can have them in medium group (label 1 )  over a period of time. Similarly we can perform this for medium customers (label 1)  and try to have them in high group (label 2) over a period of time .
# 
# 
# 
# 3. Customers in low group ( label 0 ) make the most number of customer care calls, these customers can be told about different offers to try and move them to  medium group over a period of time .
# 
