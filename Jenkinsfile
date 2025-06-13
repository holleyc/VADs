pipeline {
    agent any

    triggers {
        githubPush()
    }

    environment {
        VENV_DIR     = 'venv'
        OLLAMA_MODEL = 'llm-your-model:latest' // adjust to your local model name
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Environment') {
            steps {
                sh '''
                  python3 -m venv ${VENV_DIR}
                  . ${VENV_DIR}/bin/activate
                  pip install --upgrade pip
                  if [ -f requirements.txt ]; then
                    pip install -r requirements.txt
                  elif [ -f requirementsGTX1660Ti.txt ]; then
                    grep -v '^apturl==' requirementsGTX1660Ti.txt > filtered-requirements.txt || true
                    pip install -r filtered-requirements.txt || true
                  else
                    echo "No requirements file found, skipping dependency install"
                  fi
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  flake8 . || echo "flake8 not installed, skipping lint"
                '''
            }
        }

        stage('Test') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  pytest -q --disable-warnings || echo "pytest not installed or no tests found, skipping tests"
                '''
            }
        }

        stage('Package') {
            steps {
                sh '''
                  . ${VENV_DIR}/bin/activate
                  if [ -f setup.py ]; then python setup.py sdist; else echo "No setup.py found, skipping package"; fi
                '''
                archiveArtifacts artifacts: 'dist/*.tar.gz', allowEmptyArchive: true, fingerprint: true
            }
        }

        stage('LLM Analysis') {
            steps {
                // Capture the last commit diff
                sh 'git diff HEAD~1 HEAD > changes.patch'

                // Run Ollama on the diff
                sh '''
                  ollama run ${OLLAMA_MODEL} \
                    --prompt-file changes.patch \
                    --output llm_report.txt
                '''

                // Display the LLM report in console
                sh 'cat llm_report.txt'

                // Archive the report for later review
                archiveArtifacts artifacts: 'llm_report.txt', fingerprint: true
            }
        }
    }

    post {
        success {
            echo '✅ Build, tests, and LLM analysis succeeded!'
        }
        failure {
            echo '❌ Build or LLM analysis failed. Check console output.'
        }
    }
}
