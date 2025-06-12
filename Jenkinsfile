pipeline {
  agent any

  stages {
    stage('Checkout') {
      steps {
        git url: 'https://github.com/your-org/your-python-repo.git', branch: 'main'
      }
    }
    stage('Setup venv') {
      steps {
        sh 'python3 -m venv venv'
        sh 'source venv/bin/activate'
      }
    }
    stage('Install Dependencies') {
      steps {
        sh 'venv/bin/pip install -r requirements.txt'
      }
    }
    stage('Test') {
      steps {
        sh 'venv/bin/pytest --junitxml=results.xml'
      }
      post {
        always {
          junit 'results.xml'
        }
      }
    }
  }
}
