# GPU-FIM-Study
With recent advancements in GPU computing, many new Algorithms have been developed for the processing of large volumes of data. One goal of data processing may be Frequency Item Mining (FIM). This is when commonly associated items are mined in transactional databases. If you have ever seen the _Frqeuently Purchased Together_
section on online stores, that is created using a FIM. I recently wanted to benchmark the memory consumption and runtime performance of these multithreaded GPU algorithms. 
This is the repo for a collection of GPU-Based Frequency Itemset Mining Algorithms. It is the location where the recoded versions of these algorithms will be hosted 

This is a study focused on benchmarking and testing popular Frequent Itemset algorithms. Algorithms were pulled from IEEE Xplore & sciencedirect

# Standardized Testing
Each algorithm is run on the same exact datasets. For now the datasets we are including are 
- T10I4D100K (first 10,000 lines) for testing/debugging purposes 
- T10I4D100K

# How does this study work?
The goal is to recode all of the algorithms from a list of around 20 algorithms. I have tried to modularize the code as much as possible, providing opportunities for those working on this project to re-use code that is already written. Once all of the algorithms are completed, we will run them on standardized datasets. 
