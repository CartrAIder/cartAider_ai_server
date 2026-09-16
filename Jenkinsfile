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
        AI_HEALTH_URL = 'http://fastapi:8000/healthz'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('AI Test Image') {
            steps {
                sh 'docker build --target test --tag cartgate-ai-test:$BUILD_NUMBER .'
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

                        docker build --tag "cartgate-ai:$BUILD_NUMBER" .
                        docker rm --force cartgate-ai-server 2>/dev/null || true
                        docker run --detach \
                            --name cartgate-ai-server \
                            --network cartAider-network \
                            --network-alias fastapi \
                            --env-file "$AI_ENV_FILE" \
                            --mount "type=bind,src=$CARTGATE_MODEL_HOST_DIR,dst=/models,readonly" \
                            --restart unless-stopped \
                            "cartgate-ai:$BUILD_NUMBER"
                    '''
                }
            }
        }

        stage('Health Check') {
            steps {
                retry(24) {
                    sleep time: 5, unit: 'SECONDS'
                    timeout(time: 5, unit: 'SECONDS') {
                        sh 'curl --fail --silent --show-error "$AI_HEALTH_URL" > /dev/null'
                    }
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
