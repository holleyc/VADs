pipeline {
    agent any

    // Trigger the pipeline on GitHub pushes (requires GitHub plugin)
    triggers {
        githubPush()
    }

    environment {
        // Use a Python virtual environment
        VENV_DIR = 'venv'
    }

    stages {
        stage('Checkout') {
            steps {
                // Clone the repository based on the Jenkinsfile in SCM
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                // Create and activate virtualenv, install dependencies
                sh 'python3 -m venv ${VENV_DIR}'
                sh "source ${VENV_DIR}/bin/activate && pip install --upgrade pip"
                sh "source ${VENV_DIR}/bin/activate && pip install -r requirements.txt"
            }
        }

        stage('Lint') {
            steps {
                // Example lint step (optional)
                sh "source ${VENV_DIR}/bin/activate && flake8 ."
            }
        }

        stage('Test') {
            steps {
                // Run tests (if you have pytest or other tests)
                sh "source ${VENV_DIR}/bin/activate && pytest -q --disable-warnings"
            }
        }

        stage('Package') {
            steps {
                // Example packaging or artifact creation
                sh "source ${VENV_DIR}/bin/activate && python setup.py sdist"
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
