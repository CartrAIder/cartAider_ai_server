pipeline {
    agent {
        label 'docker_agent'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
        skipDefaultCheckout(true)
    }

    environment {
        L4T_VERSION = 'r35.4.1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('AI Test Image') {
            steps {
                sh 'docker build --build-arg L4T_VERSION=$L4T_VERSION --target test --tag cartgate-ai-test:$BUILD_NUMBER .'
            }
        }

        stage('Docker Deploy') {
            steps {
                withCredentials([
                    file(credentialsId: 'cartgate-ai-production-env', variable: 'AI_ENV_FILE')
                ]) {
                    sh '''
                        set +x
                        set -eu

                        while IFS= read -r line || [ -n "$line" ]; do
                            case "$line" in
                                ''|'#'*) continue ;;
                            esac
                            key="${line%%=*}"
                            value="${line#*=}"
                            export "$key=$value"
                        done < "$AI_ENV_FILE"

                        : "${CARTGATE_MODEL_HOST_DIR:?CARTGATE_MODEL_HOST_DIR must point to the model bundle}"
                        scripts/validate_model_bundle.sh "$CARTGATE_MODEL_HOST_DIR"

                        docker build \
                            --build-arg L4T_VERSION="$L4T_VERSION" \
                            --tag "cartgate-ai:$BUILD_NUMBER" .
                        sh scripts/deploy_with_rollback.sh \
                            "cartgate-ai:$BUILD_NUMBER" \
                            "$AI_ENV_FILE" \
                            "$CARTGATE_MODEL_HOST_DIR"
                    '''
                }
            }
        }
    }

    post {
        failure {
            sh 'docker logs --tail 200 cartgate-ai-server 2>/dev/null || true'
        }
        cleanup {
            cleanWs()
        }
    }
}
