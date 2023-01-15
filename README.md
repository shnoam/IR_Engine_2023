# IR_Engine_2023

In this project we created an IR engine on Wikipedia , based on inverted index in order to address given queries and return the best matched reuslt as possible .  

In order to be able to process and store Wikipedia's huge amount of data , we used the GCP ( Google Cloud Platform ) .GCP provided us the ability to use several machines at the same time as well as an appropriate platform to process our data using Spark and RDD . 

firstly , we built an inverted index on each of the title , body and anchor text and stored it on a GCP bucket . 
secondly , we created functions that help us to rank and evaluate the results .

files :
1. Backend_create_index_gcp.ipynb - process and manipulate the data. And building of the inverted index object and assigning each of the Title , Body and anchor text attributes and the files were created .
2. inverted_index_gcp.py - consists all attributes and methods of each of the used classes in our project ( file Reader , Writer , Inverted Index , etc .)
3. search_frontend.py - Used to run the Flask app ( the queries ) and contains most of the search and rank functions we used . 
4. BM_25_from_index.py - Implimentation of the BM25 method to calculate similarity.
5. run_frontend_in_gcp.sh - creation info of the GCP instance to query the engine .
6. files_from_bucket.docx  - stored files from bucket
