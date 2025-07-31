class MNISTClassifier {
            constructor() {
                this.session = null;
                this.isDrawing = false;
                this.lastX = 0;
                this.lastY = 0;
                this.canvas = null;
                this.ctx = null;
                this.autoPredictTimeout = null;
                
                this.initializeCanvas();
                this.bindEvents();
            }

            initializeCanvas() {
                this.canvas = document.getElementById('drawingCanvas');
                this.ctx = this.canvas.getContext('2d');
                
                // Configuration du canvas
                this.ctx.strokeStyle = '#000';
                this.ctx.lineWidth = 12;
                this.ctx.lineCap = 'round';
                this.ctx.lineJoin = 'round';
                
                // Fond blanc
                this.clearCanvas();
            }

            bindEvents() {
                const clearBtn = document.getElementById('clearBtn');
                const loadModelBtn = document.getElementById('loadModelBtn');

                clearBtn.addEventListener('click', () => this.clearCanvas());
                loadModelBtn.addEventListener('click', () => this.loadModel());

                // Événements de dessin - souris
                this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
                this.canvas.addEventListener('mousemove', this.draw.bind(this));
                this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
                this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));

                // Événements de dessin - tactile
                this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
                this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
                this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
            }

            handleTouch(e) {
                e.preventDefault();
                const rect = this.canvas.getBoundingClientRect();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent(
                    e.type === 'touchstart' ? 'mousedown' : 
                    e.type === 'touchmove' ? 'mousemove' : 'mouseup',
                    {
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    }
                );
                this.canvas.dispatchEvent(mouseEvent);
            }

            startDrawing(e) {
                this.isDrawing = true;
                [this.lastX, this.lastY] = this.getMousePos(e);
            }

            draw(e) {
                if (!this.isDrawing) return;
                
                const [currentX, currentY] = this.getMousePos(e);
                
                this.ctx.beginPath();
                this.ctx.moveTo(this.lastX, this.lastY);
                this.ctx.lineTo(currentX, currentY);
                this.ctx.stroke();
                
                [this.lastX, this.lastY] = [currentX, currentY];
                
                // Prédiction automatique avec délai
                this.scheduleAutoPrediction();
            }

            stopDrawing() {
                if (!this.isDrawing) return;
                this.isDrawing = false;
                this.scheduleAutoPrediction();
            }

            scheduleAutoPrediction() {
                if (this.autoPredictTimeout) {
                    clearTimeout(this.autoPredictTimeout);
                }
                this.autoPredictTimeout = setTimeout(() => {
                    if (this.session) {
                        this.predict();
                    }
                }, 500);
            }

            getMousePos(e) {
                const rect = this.canvas.getBoundingClientRect();
                return [
                    e.clientX - rect.left,
                    e.clientY - rect.top
                ];
            }

            clearCanvas() {
                this.ctx.fillStyle = '#fff';
                this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                this.resetPrediction();
            }

            resetPrediction() {
                document.getElementById('predictionResult').textContent = 'Dessinez un chiffre pour commencer';
                for (let i = 0; i < 10; i++) {
                    document.getElementById(`conf-${i}`).style.height = '0%';
                    document.getElementById(`text-${i}`).textContent = '0%';
                }
            }

            async loadModel() {
                const status = document.getElementById('status');
                const loadBtn = document.getElementById('loadModelBtn');
                
                try {
                    status.className = 'status loading';
                    status.textContent = 'Chargement du modèle ONNX...';
                    loadBtn.disabled = true;

                    // Vous devez remplacer ce chemin par l'URL de votre modèle ONNX
                    const modelUrl = 'mnist_model.onnx'; // Placez votre fichier .onnx à côté du HTML
                    
                    this.session = await ort.InferenceSession.create(modelUrl);
                    
                    status.className = 'status ready';
                    status.textContent = 'Modèle chargé avec succès ! Vous pouvez maintenant dessiner.';
                    loadBtn.textContent = 'Modèle chargé';
                    
                } catch (error) {
                    console.error('Erreur lors du chargement du modèle:', error);
                    status.className = 'status error';
                    status.textContent = `Erreur: ${error.message}. Assurez-vous que le fichier mnist_model.onnx est accessible.`;
                    loadBtn.disabled = false;
                }
            }

            preprocessImage() {
                // Créer un canvas temporaire de 28x28
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = 28;
                tempCanvas.height = 28;
                const tempCtx = tempCanvas.getContext('2d');
                
                // Redimensionner l'image à 28x28
                tempCtx.drawImage(this.canvas, 0, 0, 28, 28);
                
                // Obtenir les données d'image
                const imageData = tempCtx.getImageData(0, 0, 28, 28);
                const data = imageData.data;
                
                // Convertir en niveaux de gris et normaliser
                const input = new Float32Array(28 * 28);
                for (let i = 0; i < 28 * 28; i++) {
                    // Convertir RGBA en niveau de gris (inverser car MNIST utilise blanc sur noir)
                    const r = data[i * 4];
                    const g = data[i * 4 + 1];
                    const b = data[i * 4 + 2];
                    const gray = (r + g + b) / 3;
                    
                    // Normaliser selon les paramètres d'entraînement MNIST
                    // Inverser pour avoir noir sur blanc comme MNIST
                    const normalized = (255 - gray) / 255.0;
                    // Appliquer la normalisation utilisée pendant l'entraînement
                    input[i] = (normalized - 0.1307) / 0.3081;
                }
                
                return input;
            }

            async predict() {
                if (!this.session) {
                    alert('Veuillez d\'abord charger le modèle !');
                    return;
                }

                try {
                    // Préprocesser l'image
                    const inputData = this.preprocessImage();
                    
                    // Créer le tensor d'entrée
                    const inputTensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
                    
                    // Faire la prédiction
                    const results = await this.session.run({ input: inputTensor });
                    const output = results.output.data;
                    
                    // Convertir log softmax en probabilités
                    const probabilities = this.softmax(Array.from(output));
                    
                    // Trouver la prédiction
                    const predictedClass = probabilities.indexOf(Math.max(...probabilities));
                    const confidence = Math.max(...probabilities);
                    
                    // Afficher les résultats
                    this.displayResults(predictedClass, confidence, probabilities);
                    
                } catch (error) {
                    console.error('Erreur lors de la prédiction:', error);
                    document.getElementById('predictionResult').textContent = 'Erreur lors de la prédiction';
                }
            }

            softmax(logits) {
                const maxLogit = Math.max(...logits);
                const scores = logits.map(l => Math.exp(l - maxLogit));
                const sum = scores.reduce((a, b) => a + b);
                return scores.map(s => s / sum);
            }

            displayResults(predictedClass, confidence, probabilities) {
                // Afficher la prédiction principale
                const resultElement = document.getElementById('predictionResult');
                resultElement.innerHTML = `
                    <span style="font-size: 1.2em;">Prédiction:</span><br>
                    <span style="color: #4ade80; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">${predictedClass}</span>
                    <br>
                    <span style="font-size: 0.6em; color: #666;">Confiance: ${(confidence * 100).toFixed(1)}%</span>
                `;

                // Mettre à jour les barres de confiance
                probabilities.forEach((prob, i) => {
                    const percentage = (prob * 100).toFixed(1);
                    const fillElement = document.getElementById(`conf-${i}`);
                    const textElement = document.getElementById(`text-${i}`);
                    
                    fillElement.style.height = `${prob * 100}%`;
                    textElement.textContent = `${percentage}%`;
                    
                    // Mettre en évidence la prédiction principale
                    if (i === predictedClass) {
                        fillElement.style.background = 'linear-gradient(180deg, #fbbf24, #f59e0b)';
                        textElement.style.fontWeight = 'bold';
                    } else {
                        fillElement.style.background = 'linear-gradient(180deg, #4ade80, #22c55e)';
                        textElement.style.fontWeight = 'normal';
                    }
                });
            }
        }

        // Initialiser l'application
        document.addEventListener('DOMContentLoaded', () => {
            const classifier = new MNISTClassifier();
            
            // Essayer de charger le modèle automatiquement
            setTimeout(() => {
                classifier.loadModel();
            }, 1000);
        });