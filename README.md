<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/exams/assets/119248312/66273639-3ace-4fa8-b780-2ba2cc9f3375"/> 

# Exams
  
<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/exams)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/exams)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/exams.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/exams.png)](https://supervise.ly)

</div>

# Overview

This application allows you to create and manage labeling exams. 

# How To Run

**Step 1:** Run the application from the ecosystem.

**Step 2:** Wait until the app is started.

Once the app is started, new task appear in workspace tasks. Wait for the message `Application is started ...` and then press `Open` button.

**Step 3:** Open the app.

After you open the app you will see a table with all created exams.

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/ef696b96-4152-459e-a7f7-693cef6654d0" width="600"/></p>


# App features

### **Create a new Exam**

You can create a new exam by pressing `Create Exam` button. You will see a form where you can specify the name of the exam, select a benchmark dataset, classes and tags to annotate and other parameters and assign a person. After you press `Create` button, the exam will be created and you will be redirected to the exam page. When the exam is created, a labeling job for each user will be created.

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/7448d8f2-4f17-45d3-be56-eab0e07f954a" width="600"/></p>

### **See a report**

You can see a report for the exam results of any person by pressing the "show report" button in the table.

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/18662770-b37c-41cf-b983-6ff7d6a8fa74" width="600"/></p>


### **Start new attempt**

You can initiate a new attempt for a user by pressing the "new attempt" button in the table. A new labeling job will be created for the examinee and the old one will be deleted.

### **Delete an exam**

To delete an exam, you need to delete the workspace, associated with the exam. The workspace has a name starting with "Exam: <exam name>"

# How it works

Each exam is a workspace. The workspace contains the benchmark project which is a copy of the source project which you selected when creating an exam. Each User attempt is a project in the given workspace. For each attempt, a new labeling job is also created. After the labeling job is submitted, the report for the attempt can be generated. When the report of the attempt is created, report results are saved as a .json file in the "exam\_data/{attempt_project_id}" folder of Team Files and a "difference" project is created in the workspace. Such projects have names ending with "\_DIFF". The difference project contains annotations with pixel differences between the benchmark project and the attempt project annotations. If you want to re-generate the report, you need to delete the report .json file from your Team Files.

# What Report metrics mean

### Table **Objects counts per class**
- **GT Objects** - Total number of objects of the given class on all images in the benchmark dataset
- **Labeled Objects** - Total number of objects of the given class on all images in the exam dataset
- **Recall(Matched objects)** - Number of correctly labeled objects of the given class divided by the total number of objects of the given class on all images in the benchmark dataset.
- **Precision** - Number of correctly labeled objects of the given class divided by the total number of objects of the given class on all images in the exam dataset
- **F-measure** - Harmonic mean of Recall and Precision
- **Objects Score (average F-measure)** - Average of F-measures of all classes

### Table **Geometry quality**
- **Pixel Accuracy** - Number of correctly labeled or unlabeled pixels on all images divided by the total number of pixels
- **IOU** - Intersection over union. Sum of intersection areas of all objects on all images divided by the sum of union areas of all objects on all images for the given class
- **Geometry Score (average IoU)** - Average of IOUs of all classes

### Table **Tags**
- **GT Tags** - Total number of tags of all objects on all images in the benchmark dataset
- **Labeled Tags** - Total number of tags of all objects on all images in the exam dataset
- **Precision** - Number of correctly labeled tags divided by the total number of tags of all objects on all images in the exam dataset
- **Recall** - Number of correctly labeled tags divided by the total number of tags of all objects on all images in the benchmark dataset
- **F-measure** - Harmonic mean of Recall and Precision
- **Tags Score** - Average of F-measures of all tags

### Table **Report per image**
- **Objects Score** - F-measure of objects of any class on the given image
- **Objects Missing** - Number of objects of any class on the given image that are not labeled
- **Objects False Postitve** - Number of objects of any class on the given image that are labeled but not present in the benchmark dataset
- **Tags Score** - F-measure of tags of any object on the given image
- **Tags Missing** - Number of tags of any object on the given image that are not labeled
- **Tags False Positive** - Number of tags of any object on the given image that are labeled but not present in the benchmark dataset
- **Geometry Score** - Intersection over union for all objects of any class on the given image
- **Overall Score** - Average of Objects Score, Tags Score and Geometry Score
