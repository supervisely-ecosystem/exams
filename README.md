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

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/0d650534-d779-4595-99fc-57f76d57c7d0" width="600"/></p>

**Step 4**:** Create a new Exam.

You can create a new exam by pressing `Create Exam` button. You will see a form where you can specify the name of the exam, select a benchmark dataset, classes and tags to annotate and other parameters and assign a person. After you press `Create` button, the exam will be created and you will be redirected to the exam page. When the exam is created, a new labeling job for each user will be created.

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/c5c1381c-a6b2-4aef-823a-e74700227791" width="600"/></p>

**Step 5:** See a report.

You can see a report for any person by pressing the "show report" button in the table. If the examinee needs another try, you can start it by clicking on the "new attempt" button. A new labeling job will be created for the examinee and the old one will be deleted.

<p align="center"><img src="https://github.com/supervisely-ecosystem/exams/assets/61844772/fe061670-0273-41b3-a4a9-fec1fe5959e4" width="600"/></p>

**Step 6:** Delete an exam
To delete an exam, you need to delete the workspace, associated with the exam. The workspace has a name starting with "Exam: <exam name>"
