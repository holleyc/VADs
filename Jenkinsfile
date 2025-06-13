pipeline {
    agent any

    // Trigger the pipeline on GitHub pushes (requires GitHub plugin)
    triggers {
        githubPush()
    }

    environment {
        // Virtual environment directory
        VENV_DIR = 'venv'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                // Create virtualenv and install dependencies in one shell session
                sh '''
                  python3 -m venv ${VENV_DIR}
                  . ${VENV_DIR}/bin/activate
                  pip install --upgrade pip
                  pip install -r requirements.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  flake8 .
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  pytest -q --disable-warnings
                '''
            }
        }

        stage('Package') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  python setup.py sdist
                '''
                archiveArtifacts artifacts: 'dist/*.tar.gz', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Build succeeded!'
        }
        failure {
            echo 'Build failed. Check the console output.'
        }
    }
}
