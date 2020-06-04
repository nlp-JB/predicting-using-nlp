# Natural Language Processing: Predicting Programing Languages
### The purpose of this project is to use various nlp techniques and classification modeling strategies to best prredict the programing language of a repository on Github using the contents of each repo's README.  

# Deliverables 
### The team was asked to provide:
> - A jupyter notebook containing the analysis and modeling
> - A presentation slide deck highlighting the major takeaways of the project

# Executive Summary
### Key Takeaways
- We successfully created a model that performed better than our initial baseline
- Ultimately our best model performed at 57% on test data
- Additionally, we provided a function that takes in a repo and predicts the programming language 
- The key to an improved model is noise reduction

# Data Dictionary

| Feature          | Count/Datatype        | Description                                                                          |
|------------------|-----------------------|--------------------------------------------------------------------------------------|
| repo             | 298 non-null          | name of the repository                                                               |
| language         | 298 non-null          | primary language of the repository                                                   |
| gen_language     | 298 non-null          | aggregation of miscellaneous programing languages into a label called "other"        |
| readme_contents  | 298 non-null          | text present in readme file                                                          |
| num_words        | 298 non-null int64    | count of words in each readme                                                        |
| link_counts      | 298 non-null int64    | count of links in each readme                                                        |
| py_extensions    | 298 non-null int64    | count of .py file extensions in each readme                                          |
| js_extensions    | 298 non-null int64    | count of .js file extensions in each readme                                          |
| ipynb_extensions | 298 non-null int64    | count of .ipynb file extensions in each readme                                       |
| link_bins        | 298 non-null category | bins of number of links into "small", "medium", and "large"                          |
| word_bins        | 298 non-null category | bins of number of words into "small", "medium", and "large"                          |
| + columns        | 298 non-null object   | any column with a "+" afterwards is an aggregation of words similar to the root word |
| politeness       | 298 non-null object   | aggregation of words pertaining to politeness                                        |

# Pipeline 
## Aquisition
- We chose to look at 12 topics and scrape the readme files of the 30 most starred repos in each topic.
- Topics:
1. algorithm
2. bots
3. data-visualization
4. deep_learning
5. javascript
6. jupyter_notebook
7. machine_learning
8. nlp
9. python
10. repo_source
11. testing
12. Covid19

- We saved the source code of each topic page into a text file for future use and to avoid any access issues. 
- Our `create_large_df` function scrapes the repo names from each text file.
- We then applied a function called `scrape_github_data` which returns the name of the repo, language type and readme contents.
- In addition, all null rows were dropped

## Preparation 
Our `prep_readme_data` function accomplishes all of the basic clean operations including removind special characters, html tesxt and normalizes the data. We continued by creating a functions for tokenizinf, lemmatizing and removing stop words from the readme contents.

We also added a number of new features to make modeling easier. 
- General languages: grouping all the popular languages together, and adding the singletons into an 'other' category
- Number of individual words
- Number of unique words
- Number of links
- Counts of each .py, .js & .ipynb extensions

This file can be saved as a .csv for easy access and to save time.

**The data was then split and scaled using libraries from SKlearn. Random seeds were set for reproducibility.**
### Hypothesis testing

- We formed two crucial hypothises early on. The first is based on the number of links present in a readme. We Thought that perhaps more popular languages had more reference materials that made thier way into readme contents.
- Secondly, we thought longer readme files may have some bearing on the ability to classify the programming language as some communities surrounding specific languages are more inclined to document than others.

The results of these hypothesis tests are located in the Hypothesis testing section of the Walkthrough notebook. 

## Exploration

The explore stage revealed

## Modeling

The team chose three classification models to work with:
 > - Logistic Regression
 > - Random Forest
 > - K Nearest Neighbors

 To tweak the hyperparameters of each model, simply access them in the modeling.py file and adjust them as desired. They are currently set to attain optimal results
## Evaluation
- To start, a baseline model was created by simply predicting the most common programing language (Python)

- Each model was then evaluated based on accuracy, precisioin and recall.
- For the purposes of this project, we chose the model that produced the highest accuracy as we wanted to maximize predicting the right language when so many others are present in the corpus.

 > -  *Our __best__ performing model on training data was the Random Forest Model with 95% accuray and it was deployed on the test data. Final model performance was 57% accuracy.*

## Conclusion
- The model struggled to predict programming languages that appeared less frequently in our corpus of documents. 
- Natural language processesing appears to be especially prone to overfitting though with some recursive fine-tuning, it can be used to correctly classify documents according to thier programming language.

## Technical Skills and Libraries Used
- Python
- Web scraping
- Hypothesis testing
- Beautiful Soup
- Natural Language Toolkit
- Regex
- SKlearn
- Canva
- Matplotlib
- Stats


**Note: To reproduce this notebook you will need your own env.py file storing your github username and unique github personal access token. Clone the repo. All other modules, functions and text files are included in the repository.**

To view our presentation of this project, [click here](https://www.canva.com/design/DAD-GzRcYlA/sIfOlRBWbqHXU96QWgMvlQ/view?utm_content=DAD-GzRcYlA&utm_campaign=designshare&utm_medium=link&utm_source=sharebutton)




