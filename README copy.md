
## Challenge 1: Deploy an API in Azure


## Overview 

🎯 **Exercise objectives**:

* Introduce participants to building a python package ready for deployment.
* Guide participants through the process of deploying an API to Azure with FastAPI and GitHub.
* Perform code tests with pytest and check they run in GitHub Actions.
* Deploy an API using CI/CD principles to Azure.

🔧 **Tools**:

* **VS Code** for interactive coding
* **Fast API** to build REST APIs
* **joblib** to save/load predictive models
* **Azure** to deploy APIs and Web Apps
* **GitHub** to host the code of your Apps


---
## 📃 Instructions

In this challenge, you will learn how to deploy a machine learning model to predict wine quality with FastAPI. By the end of this exercise, you will have a clear understanding of how to deploy your model with an API endpoint with a MLOps approach.

### 1. Explore your API and test it locally

- Open `VS Code` with the `fastapi-env` environment and open the exercise folder. 

- Open the file `main.py` and verify that the code load the model correctly. 

- Then, check that you have the right features and types in the endpoint to make the inference. Note that we implemented the `pydantic` Model to process the features.

- Go to Environments in the Anaconda Navigator and click the green play icon (▶️) next to your `fastapi-env` environment.


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-21 171227.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMmFIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--181032820c2d01d6b445b339f1db45993b80b2d1/Capture%20d'%C3%A9cran%202024-09-21%20171227.png" />
</figure>

- A terminal will open with your environment activated. Go to your challenge folder. Then, run the command to launch the API server:

  ```bash
  cd # Gets back to the $HOME directory
  cd Documents/GitHub/ai-integration-se-challenges/02-AI-Integration-in-Software/01-Deploy-API-in-Azure
  fastapi dev main.py
  ```

- Open your browser and go to the localhost address indicated in your terminal: `http://127.0.0.1:8000/docs`. This is the documentation page of your API where you can try if it works. To do so, open the predict endpoint and use the button `Try it out`. Then, enter some values for the features and execute the endpoint. You should obtain the prediction.

- Now, let's try to use the API with code. To do so, Open the file `test.py` in VS code (using the fastapi-env) and run the file. You should obtain a predicted value and a predicted probability.

- If you were able to open the Fast API Interface and make some predictions, then you've successfully exposed your model. 

- Close your browser. Then, in the terminal, use the keyboard shorcut `Ctrl + C` to close Fast API. Close the terminal.


---
### 2. Build your local Python package 

- Create a new local repository using the same API and code as before. For instance, you can create it at this location: `Documents/GitHub/yourwineapi`. Use an unique name for each new app and without spaces, for instance your name initials, if your name is `John Doe`, then you can name your app `jdwineapi`.

- Use GitHub Desktop to initialize this local repository with Git. To do so, select `Create new repository`, then select your folder location and give the same name as your folder.  

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 183118.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM2lIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--94d3217bf9f08e3bd9520f686723620c6709446e/Capture%20d'%C3%A9cran%202024-09-22%20183118.png" />
</figure>


<figure style="width: 60%">
  <img alt="Capture d'écran 2024-09-22 183210.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM21IQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--bbe47a39b1d12ef05940f435ebe5b1fec1a4516b/Capture%20d'%C3%A9cran%202024-09-22%20183210.png" />
</figure>

- Copy and paste the file `main.py` and the model in joblib format in the new repertory. Then, you can commit your changes in GitHub Desktop with a _commit message_ in the bottom left form (for instance `First commit`).

- Create and save a new file `requirements.txt` with the python libraries installed in your environment and that we need to run your API: 

  ```python
  pip>=9
  setuptools>=26
  wheel>=0.29
  pandas
  pytest
  coverage
  numpy
  httpx
  scikit-learn
  joblib
  memoized-property
  gcsfs
  termcolor
  uvicorn
  fastapi
  pydantic
  ```

- Create and save a new file `setup.sh` with the starting command to launch the API server: 

  ```python
  gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
  ```

Note that main is the name of your file containing the FastAPI main code. When Azure Web App will start the server, it is going to use this command to launch the API.

- Create and save a new file `setup.py` to define your python package: 

  ```python
  from setuptools import find_packages
  from setuptools import setup

  with open('requirements.txt') as f:
      content = f.readlines()
  requirements = [x.strip() for x in content if 'git+' not in x]

  setup(name='yourwineapi',
        version="1.0",
        description="Project Red Wine Quality Prediction API",
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements)
  ```

- Your local repertory `Documents/GitHub/yourwineapi` should contain the following files: 

  ```
  yourwineapi
  │   model.joblib
  │   main.py
  │   requirements.txt
  │   setup.py
  │   setup.sh
  ```

- Save and commit this checkpoint in your GitHub Desktop.

---
### 3. Build and test your code with pytest

In MLOps, it is important to create testing code to enable test automation for the CI/CD process. To do so, we will first evaluate the code locally with `pytest`. Then, we will activate CI/CD with GitHub Actions later. 

- Create a new folder `/yourwineapi/tests`. Then, in this new folder create an empty file `__init__.py` and `test_fast.py`.

- In the file `test_fast.py` and add the following code:

  ```python
  from fastapi.testclient import TestClient
  from main import app
  import json

  # Initialize a FastAPI client
  client = TestClient(app)

  # Test your root endpoint
  def test_read_main():
      response = client.get("/")
      assert response.status_code == 200
      assert response.json() == {"message": "Hello stranger! This API allow you to evaluate the quality of red wine. Go to the /docs for more details."}

  # Test your predict endpoint
  def test_predict():
      response = client.post(
          "/predict",
          headers={'accept': 'application/json', 'Content-Type': 'application/json'},
          json={'alcohol':9.4, 'volatile_acidity': 0.7},
      )
      assert response.status_code == 200
      assert response.json() == json.dumps({
          'prediction': 0,
          'probability': [0.7759315070304591, 0.22406849296954087]
      })
  ```

This code will allow you to test if your endpoints are working correctly with an unit testing approach. 


- Then, open an Anaconda Prompt using the `fastapi-env`. Go your new app folder `Documents/GitHub/yourwineapi` and run the following command:

  ```bash
  cd # Gets back to the $HOME directory
  cd Documents/GitHub/yourwineapi
  pytest tests/test_fast.py
  ```

You should obtain a message with 2 tests passed and 2 warnings. If not, try to install `pytest` and run the command again:

  ```bash
  pip install pytest
  pytest tests/test_fast.py
  ```

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 190754.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3FIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--91795943f3e6645cf28d6fdb0cb6b74aa85a6eb8/Capture%20d'%C3%A9cran%202024-09-22%20190754.png" />
</figure>


If you do not obtain this result, ask a TA for help.


- Finally, create a `.gitignore` file with this content:

  ```python
  .DS_Store
  __pycache__
  .ipynb_checkpoints
  .pytest_cache
  ```


- Your local repertory `Documents/GitHub/yourwineapi` should containg the following files: 

  ```
  yourwineapi
  │   model.joblib
  │   main.py
  │   requirements.txt
  │   setup.py
  │   setup.sh
  │   .gitignore
  │
  └───tests
  │   │   __init__.py
  │   │   test_fast.py
  ```

---
### 4. Create a GitHub repository and link it to your local project

- Open GitHub Desktop and in your local repertory `Documents/GitHub/yourwineapi`, you will find the option to Publish the repository to GitHub. Click on the button and follow the instructions.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 194206.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3VIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--f2aa3222fa325c92849f987f38771178c83394c2/Capture%20d'%C3%A9cran%202024-09-22%20194206.png" />
</figure>


<figure style="width: 60%">
  <img alt="Capture d'écran 2024-09-22 194510.png"  src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBM3lIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--c414b42973f16c250dc3e46135e86cef0e0896cf/Capture%20d'%C3%A9cran%202024-09-22%20194510.png" />
</figure>

- Open your GitHub Account and check that your remote repository contains all the same files and updates as your local repository. If this is not the case, ask a TA for help.



---
### 5. Setup Azure Portal and create a Plan Service for your App

- Go to [Azure Portal](https://portal.azure.com) and connect with the credentials that LeWagon provided you during the setup. 

- Check that you have acces to the `AI_Integration` Subscription 🔑:

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-21 191147.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMjJIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--500a662c4b26fa624f8b9b662e53abf1dfa01dd0/Capture%20d'%C3%A9cran%202024-09-21%20191147.png?disposition=attachment" />
</figure>

If this is not the case, ask for help. 

- Create a new resource group. You can use your name initials as follows: if your name is `John Doe`, then you can name your resource group as `jd_ai_integration`.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 195237.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMzJIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--20694eed2cc69c22fce8ffb2d02c38d534f3c83a/Capture%20d'%C3%A9cran%202024-09-22%20195237.png?disposition=attachment" />
</figure>


- Follow the instructions and select the region `France Central`. You can add some tags to identify your resource, for instance: `workshop:ai_integration`. At the end, validate the information and create the resource groum.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 195744.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMzZIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--21372e113806e1006129428eeb4db02f51acbf42/Capture%20d'%C3%A9cran%202024-09-22%20195744.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 195813.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBMytIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--d4e3b7ca5021d7abdf38d9b4a04e6f906587cb43/Capture%20d'%C3%A9cran%202024-09-22%20195813.png?disposition=attachment" />
</figure>


- Find and open your Resource Group. Click on create to open the Market Place. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 200311.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNENIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--c63973c0d8666f40e4c4b4a1bb235bcbc65c4f64/Capture%20d'%C3%A9cran%202024-09-22%20200311.png?disposition=attachment" />
</figure>


- In the search bar, find the `App Service Plan`. Click on the Create button. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 200431.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNEdIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--bda08c10b243213ecef7f83323ad462ab0e019a4/Capture%20d'%C3%A9cran%202024-09-22%20200431.png?disposition=attachment" />
</figure>

- Follow the instructions to create the Plan. Select the `AI_Integration` Subscription and then, the resource group you created. Then you can name your plan using your initials as `jdappserviceplan`. Select the `Linux` Operating System and the region `France Central`. In the pricing plan, select the `Basic B1`

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 200938.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNEtIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--85c2d04ed486c31384ffbbd2bcb11dff16908bff/Capture%20d'%C3%A9cran%202024-09-22%20200938.png?disposition=attachment" />
</figure>


- Check the information and create the `App Service Plan`.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 201433.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNE9IQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--bc22774010eafc0223419a97e5a7ad54c4789b45/Capture%20d'%C3%A9cran%202024-09-22%20201433.png?disposition=attachment" />
</figure>



---
### 6. Create an Azure Web App Service with GitHub Actions


- Go back to Azure Portal and click on `Create a new resource`. In the market place, find the `Azure Web App` service and click on create. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 201736.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNFNIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--082a3078a48a073cf6d4af5669d73e1b3a5aa897/Capture%20d'%C3%A9cran%202024-09-22%20201736.png?disposition=attachment" />
</figure>


- Follow the instructions with the following information:
- In the `Basics tab`, select the `AI_Integration` Subscription and then, the resource group you created. Give a name to the instance with the same you used for your GitHub repository, e.g. `jdwineapi`. Select the `Code` Publish mode (Note that the `Container can be used to deploy Docker Images`). Then, select the same Python version you used to build your model (in Anaconda) to the Runtime stack, e.g. `Python 3.11`. Select `Linux` as Operating System and the region `France Central`. In the Linux Plan, select the Plan App you created before, e.g.  `jdappserviceplan (B1)`.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 202028.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNFdIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--e5b847ffae55e51160635c8a38609f58b148f286/Capture%20d'%C3%A9cran%202024-09-22%20202028.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 202040.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNGFIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--39b5e68825daaa71ed7f1e9cfaa56322475a268e/Capture%20d'%C3%A9cran%202024-09-22%20202040.png?disposition=attachment" />
</figure>


- In the `Deployment tab`, enable the continuous deployment with GitHub Actions. Authorize Azure App Service to use your GitHub Account. This will open a window to log in with your GitHub account. Select the repository containing the API

<figure style="width: 60%">
  <img alt="Capture d'écran 2024-09-22 202734.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNGVIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--fdc57d32db074e04bbf42b71ccaaa2488350d119/Capture%20d'%C3%A9cran%202024-09-22%20202734.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 203129.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNGlIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--e383d3c0094aaa6122f0216571297b50f43d3737/Capture%20d'%C3%A9cran%202024-09-22%20203129.png?disposition=attachment" />
</figure>

- For the following tabs, you can keep the default values. At the end, check the information and create the Web App. 


- Open your new Web App and go to the Configuration tab. In the Startup Command write `setup.sh`.

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 204220.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNG1IQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--6120ad80d688b370ec85e614b40eaf89c1038ad8/Capture%20d'%C3%A9cran%202024-09-22%20204220.png?disposition=attachment" />
</figure>

- Go back to your GitHub repertory. You will see that Azure created a new repertory `.github/workflows` containing a yml file:

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 204617.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNHFIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--056cf129c9a3905b87d774df8376b3fd26e8139b/Capture%20d'%C3%A9cran%202024-09-22%20204617.png?disposition=attachment" />
</figure>

This file will allow you to deploy the API with a CI/CD approach. Every time you make a change locally, GitHub Actions will deploy the application in two steps: 
  * **build** the artifacts and make the tests in a virtual machine.
  * if the build step finished successfully, then, github will **deploy** the app to Azure. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 205956.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNHVIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--132fb50a04a8cc4b3f768354a9525f7ba6982b64/Capture%20d'%C3%A9cran%202024-09-22%20205956.png?disposition=attachment" />
</figure>


---
### 7. Finish the Continuous Integration with the tests.

- Go back to GitHub Desktop and use `git pull` to get the .yml file from the remote repository.

- Open the .yml file with VS Code and modify the build step. Find the comment to add the step to run tests (after installing dependencies). Then, add the last 2 lines to perform the tests. 

``` 
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      # Optional: Add step to run tests here (PyTest, Django test suites, etc.)
      - name: Test with pytest
        run: pytest tests/test_fast.py
```

- Save the File and commit the project. Then, push the project to GitHub and check the GitHub Actions. You will observe that the steps run correctly and update the Web App in Azure.


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 212449.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNHlIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--df9090514ef287e6233dcc7a6eb6cc75ca03ef6f/Capture%20d'%C3%A9cran%202024-09-22%20212449.png?disposition=attachment" />
</figure>


---
### 8. Test your API

- Go to your Web App panel and look for the `Default Domain`. Copy this url and open it in a new browser tab. You should be able to request the root endpoint of the API. If you add `/docs` to the url, you will be able to test the `/predict` endpoint as you did it locally. 

<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211814.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDJIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--ed9c79dc73c59e1ff54c08ff956b3cbcd57e22a0/Capture%20d'%C3%A9cran%202024-09-22%20211814.png?disposition=attachment" />
</figure>


<figure style="width: 80%">
  <img alt="Capture d'écran 2024-09-22 211846.png" src="https://learn.lewagon.com/rails/active_storage/blobs/redirect/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBNDZIQmc9PSIsImV4cCI6bnVsbCwicHVyIjoiYmxvYl9pZCJ9fQ==--4510164d2c2456f900838fc9c9b2e50f9f0c2e24/Capture%20d'%C3%A9cran%202024-09-22%20211846.png?disposition=attachment" />
</figure>


#### 🛠️🆘 Troubleshooting: if you have problems with your app, you might need to restart the machine in the Web App portal in Azure or redeploy the steps in GitHub Action. Just be patient, it can time several minutes to work.

- Open the `test.py` file with VS Code and modify the URL with your new Web app domain. Try the code and check that you can make predictions. 

Well done! You have deployed your first API to Azure with an MLOps approach. 

---

### Don't forget to save your work!

Save your files: File > Save. Then close all the tabs in your browser and VS Code windows. You can safely close the Anaconda Prompt.

:bulb: Don't forget to **push your code to GitHub**

1. Open GitHub Desktop.
2. It should automatically detect any file with modifications. If not, ask a TA.
3. Make sure these files are ticked, and write a _commit message_ in the bottom left form.
4. Click on the "Commit to `master`" button at the bottom of the form.
5. Click on the "Push `origin`" button at the top of the window.


---
## 🥇 Key learning points 
By the end of this exercise, you will have: 
* Learned how to deploy an API to expose your model with Azure Web App Services.
* Gained experience in using FastAPI and GitHub Actions to deploy models with a CI/CD approach.

That's it! Take a small break before diving into the next exercise.



