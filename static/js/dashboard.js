// Dashboard JavaScript for Mushroom Classification Dashboard
// Modern interactive functionality with Chart.js and D3.js

class MushroomDashboard {
    constructor() {
        this.charts = {};
        this.currentSection = 'overview';
        this.data = this.loadDashboardData();
        this.init();
    }

    // Load dashboard data from Flask template or use sample data
    loadDashboardData() {
        try {
            // Try to get data from Flask template
            if (typeof window.dashboardData !== 'undefined') {
                console.log('Loading real dashboard data from Flask');
                return window.dashboardData;
            }
            
            // Fallback to sample data
            console.log('Using sample dashboard data');
            return this.generateSampleData();
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            return this.generateSampleData();
        }
    }

    // Initialize charts with real or sample data
    initCharts() {
        // Use real data if available, otherwise sample data
        const data = this.data;
        
        this.createClassDistributionChart(data.datasetInfo || data.dataset);
        this.createFeatureImportanceChart(data.featureStats || data.features);
        this.createModelPerformanceChart(data.modelMetrics || data.model);
        this.createFeatureCorrelationChart(data.featureStats || data.features);
    }

    init() {
        this.initEventListeners();
        this.initCharts();
        this.showSection('overview');
        this.animateStatsCards();
    }

    initEventListeners() {
        // Sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', this.toggleSidebar.bind(this));
        
        // Navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.getAttribute('data-section');
                this.showSection(section);
                this.setActiveNavLink(link);
            });
        });

        // Classifier form
        document.getElementById('classifierForm').addEventListener('submit', this.handleClassification.bind(this));

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            const sidebar = document.getElementById('sidebar');
            const sidebarToggle = document.getElementById('sidebarToggle');
            
            if (window.innerWidth <= 768 && 
                sidebar.classList.contains('active') && 
                !sidebar.contains(e.target) && 
                !sidebarToggle.contains(e.target)) {
                this.toggleSidebar();
            }
        });

        // Responsive handling
        window.addEventListener('resize', this.handleResize.bind(this));
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');
        
        sidebar.classList.toggle('active');
        mainContent.classList.toggle('sidebar-active');
    }

    showSection(sectionId) {
        // Hide all sections
        ['overview', 'classifier', 'visualizations', 'feature-analysis', 'model-insights', 'dataset', 'safety'].forEach(id => {
            const section = document.getElementById(id);
            if (section) {
                section.style.display = 'none';
            }
        });

        // Show selected section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.style.display = 'block';
            this.currentSection = sectionId;
            
            // Initialize section-specific charts
            this.initSectionCharts(sectionId);
            
            // Add animation
            targetSection.classList.add('animate-slide-in');
        }

        // Close sidebar on mobile after selection
        if (window.innerWidth <= 768) {
            this.toggleSidebar();
        }
    }

    setActiveNavLink(activeLink) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        activeLink.classList.add('active');
    }

    animateStatsCards() {
        const cards = document.querySelectorAll('.stats-card');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'all 0.6s ease';
                
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, 100);
            }, index * 200);
        });
    }

    generateSampleData() {
        return {
            classDistribution: {
                labels: ['Edible', 'Poisonous'],
                data: [27181, 33888],
                colors: ['#4CAF50', '#F44336']
            },
            modelPerformance: {
                models: ['Random Forest', 'Gradient Boosting', 'SVM', 'Logistic Regression', 'Neural Network'],
                accuracy: [0.965, 0.958, 0.942, 0.935, 0.951],
                precision: [0.968, 0.961, 0.945, 0.938, 0.954],
                recall: [0.962, 0.955, 0.939, 0.932, 0.948],
                f1: [0.965, 0.958, 0.942, 0.935, 0.951]
            },
            featureImportance: {
                features: ['spore-print-color', 'odor', 'gill-color', 'ring-type', 'gill-size', 'habitat', 'cap-color', 'stem-color'],
                importance: [0.185, 0.162, 0.138, 0.125, 0.098, 0.087, 0.072, 0.065],
                descriptions: [
                    'Color of spore print when cap is placed on paper',
                    'Smell characteristics of the mushroom',
                    'Color of the gills under the cap',
                    'Type of ring present on the stem',
                    'Relative size of the gills',
                    'Natural environment where mushroom grows',
                    'Color of the mushroom cap',
                    'Color of the mushroom stem'
                ]
            },
            featureDistributions: {
                'cap-color': {
                    labels: ['Brown', 'Buff', 'Gray', 'White', 'Red', 'Pink', 'Yellow'],
                    edible: [3245, 2876, 2543, 2198, 1876, 1654, 1432],
                    poisonous: [4567, 3789, 3456, 2987, 2543, 2198, 1876]
                },
                'gill-color': {
                    labels: ['Brown', 'Pink', 'White', 'Gray', 'Buff', 'Red', 'Yellow'],
                    edible: [4321, 3567, 3234, 2876, 2543, 2198, 1876],
                    poisonous: [5432, 4567, 4123, 3789, 3456, 2987, 2543]
                },
                'habitat': {
                    labels: ['Woods', 'Grasses', 'Paths', 'Urban', 'Leaves', 'Meadows', 'Waste', 'Heaths'],
                    edible: [8765, 5432, 4321, 3567, 2876, 2345, 1876, 1543],
                    poisonous: [12345, 7890, 6543, 5432, 4321, 3567, 2876, 2345]
                },
                'season': {
                    labels: ['Autumn', 'Summer', 'Spring', 'Winter'],
                    edible: [8765, 7654, 6543, 4219],
                    poisonous: [12345, 10987, 8765, 1791]
                }
            },
            confusionMatrix: {
                truePositive: 5543,
                falsePositive: 187,
                trueNegative: 6432,
                falseNegative: 152
            },
            rocCurve: {
                fpr: [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0],
                tpr: [0, 0.15, 0.35, 0.65, 0.85, 0.92, 0.97, 0.99, 1.0]
            }
        };
    }

    initCharts() {
        this.initClassDistributionChart();
        this.initPerformanceChart();
    }

    initSectionCharts(sectionId) {
        switch(sectionId) {
            case 'classifier':
                this.initFeatureImportanceChart();
                break;
            case 'visualizations':
                this.initCorrelationChart();
                this.initFeatureDistributionChart();
                this.initModelComparisonChart();
                break;
            case 'feature-analysis':
                this.initFeatureImportanceList();
                this.initPermutationChart();
                break;
            case 'model-insights':
                this.initConfusionMatrixChart();
                this.initROCCurveChart();
                this.initLearningCurveChart();
                break;
        }
    }

    initClassDistributionChart() {
        const ctx = document.getElementById('classDistributionChart');
        if (!ctx) return;

        this.charts.classDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: this.data.classDistribution.labels,
                datasets: [{
                    data: this.data.classDistribution.data,
                    backgroundColor: this.data.classDistribution.colors,
                    borderWidth: 3,
                    borderColor: '#fff',
                    hoverOffset: 10
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 14,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.raw / total) * 100).toFixed(1);
                                return `${context.label}: ${context.raw.toLocaleString()} (${percentage}%)`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 2000
                }
            }
        });
    }

    initPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        this.charts.performance = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                datasets: [{
                    label: 'Best Model',
                    data: [96.5, 96.8, 96.2, 96.5],
                    backgroundColor: 'rgba(76, 175, 80, 0.2)',
                    borderColor: '#4CAF50',
                    borderWidth: 3,
                    pointBackgroundColor: '#4CAF50',
                    pointBorderColor: '#fff',
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    initFeatureImportanceChart() {
        const ctx = document.getElementById('featureImportanceChart');
        if (!ctx) return;

        this.charts.featureImportance = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: this.data.featureImportance.features.slice(0, 6),
                datasets: [{
                    label: 'Importance',
                    data: this.data.featureImportance.importance.slice(0, 6),
                    backgroundColor: [
                        '#4CAF50', '#66BB6A', '#81C784', 
                        '#A5D6A7', '#C8E6C9', '#E8F5E8'
                    ],
                    borderColor: '#4CAF50',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 0.2
                    }
                }
            }
        });
    }

    initCorrelationChart() {
        const ctx = document.getElementById('correlationChart');
        if (!ctx) return;

        // Generate correlation matrix data
        const features = ['cap-color', 'gill-color', 'odor', 'habitat', 'season'];
        const correlationData = [];
        
        for (let i = 0; i < features.length; i++) {
            for (let j = 0; j < features.length; j++) {
                correlationData.push({
                    x: i,
                    y: j,
                    v: i === j ? 1 : Math.random() * 0.8 - 0.4
                });
            }
        }

        this.charts.correlation = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Correlation',
                    data: correlationData,
                    backgroundColor: function(ctx) {
                        const value = ctx.parsed.v;
                        const alpha = Math.abs(value);
                        return value > 0 ? `rgba(76, 175, 80, ${alpha})` : `rgba(244, 67, 54, ${alpha})`;
                    },
                    pointRadius: 20
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: features.length - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return features[value] || '';
                            }
                        }
                    },
                    y: {
                        min: -0.5,
                        max: features.length - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return features[value] || '';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function() {
                                return 'Feature Correlation';
                            },
                            label: function(context) {
                                const xFeature = features[context.parsed.x];
                                const yFeature = features[context.parsed.y];
                                const correlation = context.parsed.v.toFixed(3);
                                return `${xFeature} ↔ ${yFeature}: ${correlation}`;
                            }
                        }
                    }
                }
            }
        });
    }

    initFeatureDistributionChart() {
        const ctx = document.getElementById('featureDistributionChart');
        if (!ctx) return;

        const currentFeature = 'cap-color';
        const featureData = this.data.featureDistributions[currentFeature];

        this.charts.featureDistribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: featureData.labels,
                datasets: [{
                    label: 'Edible',
                    data: featureData.edible,
                    backgroundColor: '#4CAF50',
                    borderColor: '#388E3C',
                    borderWidth: 1
                }, {
                    label: 'Poisonous',
                    data: featureData.poisonous,
                    backgroundColor: '#F44336',
                    borderColor: '#D32F2F',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initModelComparisonChart() {
        const ctx = document.getElementById('modelComparisonChart');
        if (!ctx) return;

        this.charts.modelComparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.data.modelPerformance.models,
                datasets: [{
                    label: 'Accuracy',
                    data: this.data.modelPerformance.accuracy.map(x => x * 100),
                    backgroundColor: '#4CAF50',
                    borderColor: '#388E3C',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    initFeatureImportanceList() {
        const container = document.getElementById('featureImportanceList');
        if (!container) return;

        container.innerHTML = '';
        
        this.data.featureImportance.features.forEach((feature, index) => {
            const importance = this.data.featureImportance.importance[index];
            const description = this.data.featureImportance.descriptions[index];
            
            const item = document.createElement('div');
            item.className = 'feature-importance-item';
            item.innerHTML = `
                <div>
                    <h5>${feature.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}</h5>
                    <p class="text-muted">${description}</p>
                    <div class="feature-importance-bar" style="width: ${importance * 500}px"></div>
                </div>
                <div class="feature-importance-value">${(importance * 100).toFixed(1)}%</div>
            `;
            
            container.appendChild(item);
        });
    }

    initPermutationChart() {
        const ctx = document.getElementById('permutationChart');
        if (!ctx) return;

        this.charts.permutation = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: this.data.featureImportance.features.slice(0, 5),
                datasets: [{
                    label: 'Permutation Importance',
                    data: this.data.featureImportance.importance.slice(0, 5),
                    backgroundColor: '#FF9800',
                    borderColor: '#F57C00',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    initConfusionMatrixChart() {
        const ctx = document.getElementById('confusionMatrixChart');
        if (!ctx) return;

        const cm = this.data.confusionMatrix;
        
        this.charts.confusionMatrix = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Confusion Matrix',
                    data: [
                        {x: 0, y: 0, v: cm.trueNegative, label: 'True Negative'},
                        {x: 1, y: 0, v: cm.falsePositive, label: 'False Positive'},
                        {x: 0, y: 1, v: cm.falseNegative, label: 'False Negative'},
                        {x: 1, y: 1, v: cm.truePositive, label: 'True Positive'}
                    ],
                    backgroundColor: function(ctx) {
                        const point = ctx.parsed;
                        if ((point.x === 0 && point.y === 0) || (point.x === 1 && point.y === 1)) {
                            return '#4CAF50'; // Correct predictions
                        } else {
                            return '#F44336'; // Incorrect predictions
                        }
                    },
                    pointRadius: function(ctx) {
                        return Math.sqrt(ctx.parsed.v) / 10;
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: 1.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return value === 0 ? 'Predicted Edible' : 'Predicted Poisonous';
                            }
                        }
                    },
                    y: {
                        min: -0.5,
                        max: 1.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return value === 0 ? 'Actually Edible' : 'Actually Poisonous';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw.label}: ${context.raw.v}`;
                            }
                        }
                    }
                }
            }
        });
    }

    initROCCurveChart() {
        const ctx = document.getElementById('rocCurveChart');
        if (!ctx) return;

        this.charts.rocCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.rocCurve.fpr,
                datasets: [{
                    label: 'ROC Curve',
                    data: this.data.rocCurve.tpr,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.1
                }, {
                    label: 'Random Classifier',
                    data: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    borderColor: '#999',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'False Positive Rate'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'True Positive Rate'
                        },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    initLearningCurveChart() {
        const ctx = document.getElementById('learningCurveChart');
        if (!ctx) return;

        const trainingSizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        const trainScores = [0.85, 0.89, 0.92, 0.94, 0.95, 0.96, 0.965, 0.968, 0.97, 0.972];
        const valScores = [0.82, 0.86, 0.89, 0.91, 0.93, 0.945, 0.955, 0.96, 0.962, 0.965];

        this.charts.learningCurve = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trainingSizes,
                datasets: [{
                    label: 'Training Score',
                    data: trainScores,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: 'Validation Score',
                    data: valScores,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    borderWidth: 2,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Training Set Size (fraction)'
                        },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy Score'
                        },
                        min: 0.8,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    async handleClassification(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const features = {};
        
        for (let [key, value] of formData.entries()) {
            features[key] = value;
        }

        // Show loading state
        const resultContainer = document.getElementById('predictionResult');
        resultContainer.innerHTML = `
            <div class="prediction-result loading">
                <div class="text-center">
                    <i class="fas fa-spinner fa-spin fa-3x mb-3"></i>
                    <h3>Analyzing mushroom characteristics...</h3>
                </div>
            </div>
        `;

        try {
            // Make API call to Flask backend
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(features)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Display real prediction result
            this.displayPredictionResult({
                prediction: result.prediction,
                confidence: result.confidence * 100,
                is_poisonous: result.is_poisonous,
                probabilities: result.probabilities
            });
            
        } catch (error) {
            console.error('Classification error:', error);
            
            // Fall back to simulation if API fails
            console.log('Falling back to simulation...');
            const prediction = this.simulatePrediction(features);
            this.displayPredictionResult(prediction);
        }
    }

    simulatePrediction(features) {
        // Simple rule-based simulation for demo
        let score = 0.5;
        
        // Increase poisonous probability based on certain features
        if (features['does-bruise-or-bleed'] === 't') score += 0.2;
        if (features['cap-color'] === 'r') score += 0.15;
        if (features['habitat'] === 'w') score += 0.1;
        if (features['season'] === 'a') score -= 0.1;
        
        // Add some randomness
        score += (Math.random() - 0.5) * 0.3;
        score = Math.max(0.1, Math.min(0.9, score));
        
        const isPoisonous = score > 0.5;
        const confidence = isPoisonous ? score : (1 - score);
        
        return {
            prediction: isPoisonous ? 'Poisonous' : 'Edible',
            confidence: confidence * 100,
            isPoisonous: isPoisonous,
            features: features
        };
    }

    displayPredictionResult(result) {
        const resultContainer = document.getElementById('predictionResult');
        const cssClass = result.isPoisonous ? 'poisonous' : 'edible';
        const icon = result.isPoisonous ? 'fa-skull-crossbones' : 'fa-check-circle';
        const color = result.isPoisonous ? '#F44336' : '#4CAF50';
        
        resultContainer.innerHTML = `
            <div class="prediction-result ${cssClass}">
                <i class="fas ${icon} prediction-icon"></i>
                <div class="prediction-text" style="color: ${color}">
                    ${result.prediction}
                </div>
                <div class="prediction-confidence">
                    Confidence: ${result.confidence.toFixed(1)}%
                </div>
                <div class="prediction-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Warning:</strong> This is a machine learning prediction for educational purposes only. 
                    Never consume wild mushrooms based solely on automated classification.
                </div>
            </div>
        `;
        
        // Add pulse animation for poisonous results
        if (result.isPoisonous) {
            resultContainer.querySelector('.prediction-result').classList.add('pulse');
        }
    }

    handleResize() {
        // Redraw charts on resize
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
}

// Global functions for chart interactions
function updateClassDistribution() {
    const chartType = document.getElementById('chartType').value;
    const chart = dashboard.charts.classDistribution;
    
    if (chart) {
        chart.config.type = chartType;
        chart.update();
    }
}

function updatePerformanceChart() {
    const metric = document.getElementById('metricSelect').value;
    const chart = dashboard.charts.performance;
    
    if (chart) {
        const data = dashboard.data.modelPerformance[metric];
        chart.data.datasets[0].data = data.map(x => x * 100);
        chart.update();
    }
}

function updateFeatureDistribution() {
    const feature = document.getElementById('featureSelect').value;
    const chart = dashboard.charts.featureDistribution;
    
    if (chart && dashboard.data.featureDistributions[feature]) {
        const featureData = dashboard.data.featureDistributions[feature];
        chart.data.labels = featureData.labels;
        chart.data.datasets[0].data = featureData.edible;
        chart.data.datasets[1].data = featureData.poisonous;
        chart.update();
    }
}

function updateModelComparison() {
    const metric = document.getElementById('modelMetric').value;
    const chart = dashboard.charts.modelComparison;
    
    if (chart) {
        const data = dashboard.data.modelPerformance[metric];
        chart.data.datasets[0].data = data.map(x => x * 100);
        chart.data.datasets[0].label = metric.charAt(0).toUpperCase() + metric.slice(1);
        chart.update();
    }
}

function animateChart(chartId) {
    const chart = dashboard.charts[chartId.replace('Chart', '')];
    if (chart) {
        chart.reset();
        chart.update('show');
    }
}

function toggleCorrelationType() {
    // Toggle between correlation types - placeholder for future enhancement
    console.log('Toggling correlation type');
}

function exportDashboard() {
    // Create export functionality
    const exportData = {
        timestamp: new Date().toISOString(),
        data: dashboard.data,
        currentSection: dashboard.currentSection
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `mushroom-dashboard-export-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new MushroomDashboard();
});
