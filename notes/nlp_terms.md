# NLP Terms
```
POS = part of speech  
NP-chunking = noun phrase chunking  

DT = determiner
JJ = adjectives
NN = noun
VBD = verb
```

### IE (Information Extraction)
IE turns the unstructured information embedded in texts into structured data. 


### IOB (Inside, Outside, Beginning)
```
The most widespread file representation uses IOB tags:
IOB = Inside-Outside-Begininning
B = beginnning (marks beginning of chunk)
I = inside (all subsequent parts of chunk)
O = outside
```

### Named Entity
anything that can be referred to with a proper name


### NER (Named Entity Recognition)
task of detecting and classifying all the proper names mentioned in a text  
* Generic NER:  finds names of people, places and organizations that are mentioned in ordinary news texts
* practical applications:  built to detect everything  from names of genes and proteins, to names of college courses

### Reference Resolution (Coreference)
occurs when two or more expressions in a text refer to the same person or thing; they have the same referent, e.g. Bill said he would come; the proper noun Bill and the pronoun he refer to the same person, namely to Bill

### Relation Detection and Classification
find and classify semantic relations among the entities discovered in a given text

### Event Detection and Classification
find and classify the events in which the entities are participating

### Temporal Expression Detection
* tells us that our sample text contains the temporal expressions *Friday* and *Thursday*
* includes date expressions such as days of the week, months, holidays, as well as relative expressions including phrases like *two days from now* or *next year*.
* includes time:  noon, 3pm, etc.

### Temporal Analysis
over problem is to map temporal expressions onto specific calendar dates or times of day and then to use those times to situate events in time.  
