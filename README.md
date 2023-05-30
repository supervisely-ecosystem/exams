<div align="center" markdown> 

# Exams
  
<p align="center">

  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a>
</p>

[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)


</div>

## Overview

This App allows you to create and manage labeling exams. 

## How To Run

### Step 1. Run the application from the ecosystem

### Step 2. Wait until the app is started
Once the app is started, new task appear in workspace tasks. Wait for the message `Application is started ...` and then press `Open` button

### Step 3. Open the app
After you open the app you will see a table with all created exams.

<img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/0d650534-d779-4595-99fc-57f76d57c7d0" width="600"/>

### Step 4. Create new Exam
You can create a new exam by pressing `Create Exam` button. You will see a form where you can specify the name of the exam, select a benchmark dataset, classes and tags to annotate and other parameters and assign a person. After you press `Create` button, the exam will be created and you will be redirected to the exam page. When the exam is created, a new labeling job for each user will be created.

<img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/c5c1381c-a6b2-4aef-823a-e74700227791" width="600"/>

### Step 5. See a report
You can see a report for any person by pressing the "show report" button in the table. If the examinee needs another try, you can start another try by cloning the labeling job. The last created labeling job will be considered as the last try.

<img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/fe061670-0273-41b3-a4a9-fec1fe5959e4" width="600"/>

### Step 6. Delete an exam
To delete an exam, you need to delete the workspace, associated with the exam. The workspace has a name like "Exam: <exam name>"
