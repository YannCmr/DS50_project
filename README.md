# DS50_project
A project with Ilness and CNN


# Project Directory Structure

The dataset is organized for a classification task. It follows a common train test split, with further categorization by disease.

The project directory structure is as follows:

```md
DS50_project
├── 00_archives     <-- Folder for the dataset, explain underneath        
├── 10_notebooks       <-- Folder for Jupyter notebooks       
|     ├── slip_data.ipynb     <-- File for splitting the dataset smaller datasets to work with   
├── 99_Documents     <-- Folder for documents related to the project    
├── requirements.txt       <-- The requirements file for the project    
└── README.md       <-- This file explaning the project     
```


## Dataset Directory Structure
```md
archive    
├── data    
│    ├── test    
│    │   ├── Coccidiosis    
│    │   ├── Healthy    
│    │   ├── New Castle Disease    
│    │   └── Salmonella    
│    ├── train    
│    │   ├──── Coccidiosis    
│    │   │     ├── cocci.0.jpg   
│    │   │     ├── cocci.0.jpg_aug1.JPG   
│    │   │     ├── cocci.0.jpg_aug2.JPG   
│    │   │     ├── cocci.0.jpg_aug3.JPG   
│    │   │     ├── cocci.0.jpg_aug4.JPG   
│    │   │     └── cocci.0.jpg_aug5.JPG   
│    │   ├── Healthy    
│    │   ├── New Castle Disease    
│    │   └── Salmonella    
│    └── val    
│         ├── Coccidiosis    
│         ├── Healthy    
│         ├── New Castle Disease    
│         └── Salmonella    
└── data_samples    
     ├── test    
     │   ├── Coccidiosis    
     │   ├── Healthy    
     │   ├── New Castle Disease    
     │   └── Salmonella    
     ├── train    
     │   ├── Healthy    
     │   ├── New Castle Disease    
     │   └── Salmonella    
     └── val    
         ├── Coccidiosis    
         ├── Healthy    
         ├── New Castle Disease    
         └── Salmonella    
```

# venv
let's create a virtual environment and install the required packages.

```bash
python -m venv venvProjetDS50
```


Then, activate the virtual environment and install the required packages:
*For linux*
```bash
source venvProjetDS50/bin/activate
```

For Windows, the command to activate the virtual environment is slightly different:
```bash
venvProjetDS50\Scripts\activate
```

Then, install the required packages using pip:
```bash
pip install -r requirements.txt
```

# Git Tuto
A quick organisation for the git repository:
---
First, before making any changes to the code, you should **create an issue** on GitHub to describe the changes you plan to make. This helps keep track of what needs to be done and allows others to see what you're working on (don't forget to assign the issue to yourself) :
```bash
git checkout -b <branch_name> # create a new branch for your changes
```

Then you can make your changes to the code. Once you're done, you should **commit your changes** with a descriptive message:
```bash
git add . # add all changes to the staging area
git commit -m "Add a descriptive message about your changes"
```

Then, you can **push your changes** to the remote repository:
```bash
git push origin <branch_name> # push your changes to the remote repository
```

Finally, you can create a **pull request** on GitHub to merge your changes into the main branch. This allows others to review your changes before they are merged. (Nice To Have)

---
If you are working on **the wrong branch**, you can switch to the **correct branch  without deleting your work**. You can *safely* move your work to the right branch without losing anything. Here's what to do:



#### If you haven’t pushed your changes yet (only local changes):

1. **Stash or commit your changes**:
   - If you're not ready to commit yet:  
     ```bash
     git stash
     ```
   - If you're ready to commit:
     ```bash
     git add .
     git commit -m "Your message"
     ```

2. **Switch to the correct branch**:
   ```bash
   git checkout correct-branch
   ```

3. **Apply the changes**:
   - If you stashed:
     ```bash
     git stash pop
     ```
   - If you committed, you can **cherry-pick** your commit (see below).

---

#### If you already committed but not pushed:

1. **Get the commit hash**:
   ```bash
   git log
   ```
   (Copy the hash of your commit(s))

2. **Switch to the correct branch**:
   ```bash
   git checkout correct-branch
   ```

3. **Cherry-pick the commit(s)**:
   ```bash
   git cherry-pick <commit-hash>
   ```

4. **Go back to the wrong branch and remove the commit** (optional):
   If you want to clean up the wrong branch:
   ```bash
   git checkout wrong-branch
   git reset --hard HEAD~1  # or more if needed
   ```

