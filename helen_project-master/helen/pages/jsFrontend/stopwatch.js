// Configuración global
const CONFIG = {
    UPDATE_INTERVAL: 10, // actualiza cada 10ms para los milisegundos
    DIAS: ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'],
    MESES: ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
};

// Clase principal para la aplicación del cronómetro
class TimerApp {
    constructor() {
        // Elementos DOM que se necesitarán
        this.elements = {
            dateSection: $('.date-section'),
            timerDisplay: $('.timer-display'),
            milliseconds: $('.milliseconds'),
            timerIcon: $('.timer-icon'),
            totalTime: $('.total-time'),
            lapCount: $('.lap-count'),
            avgLap: $('.avg-lap'),
            startBtn: $('.start-btn'),
            resetBtn: $('.reset-btn'),
            lapBtn: $('.lap-btn'),
            lapsList: $('.laps-list'),
            loadingIndicator: $('.timer-loading')
        };

        // Variables para el cronómetro
        this.startTime = 0;
        this.elapsedTime = 0;
        this.timerInterval = null;
        this.running = false;
        this.laps = [];

        // Inicializa la aplicación para que todo funcione
        this.init();
    }

    // Llama a las funciones principales
    init() {
        this.setupEventListeners();
        this.startDateTimer();
        this.resetTimer();
        
        // Oculta el indicador de carga después de la inicialización
        setTimeout(() => {
            this.elements.loadingIndicator.fadeOut();
        }, 500);
    }

    // Configura los event listeners
    setupEventListeners() {
        // Inicia o detiene el cronómetro
        this.elements.startBtn.on('click', () => {
            if (!this.running) {
                this.startTimer();
            } else {
                this.stopTimer();
            }
        });

        // Reinicia el cronómetro
        this.elements.resetBtn.on('click', () => {
            this.resetTimer();
        });

        // Registra una vuelta
        this.elements.lapBtn.on('click', () => {
            this.addLap();
        });
           
        // Soporte para teclas
        $(document).on('keydown', (e) => {
            // Espacio para iniciar/detener
            if (e.key === ' ' || e.code === 'Space') {
                e.preventDefault();
                if (!this.running) {
                    this.startTimer();
                } else {
                    this.stopTimer();
                }
            }
            
            // L para vuelta
            if ((e.key === 'l' || e.key === 'L') && this.running) {
                this.addLap();
            }
            
            // R para reiniciar cuando esté detenido
            if ((e.key === 'r' || e.key === 'R') && !this.running) {
                this.resetTimer();
            }
        });
    }

    // Actualiza la fecha mostrada
    startDateTimer() {
        this.updateDate();
        setInterval(() => this.updateDate(), 60000); // actualiza cada minuto
    }

    updateDate() {
        const date = new Date();
        const dateString = `${CONFIG.DIAS[date.getDay()]}, ${date.getDate()} de ${CONFIG.MESES[date.getMonth()]} de ${date.getFullYear()}`;
        this.elements.dateSection.html(dateString);
    }

    // Inicia el cronómetro
    startTimer() {
        if (!this.running) {
            this.running = true;
            const now = Date.now();
            this.startTime = now - this.elapsedTime;
            
            this.timerInterval = setInterval(() => {
                this.updateElapsedTime();
            }, CONFIG.UPDATE_INTERVAL);

            // Actualiza la UI
            this.elements.startBtn.html('<i class="bi bi-pause-fill"></i><span>Detener</span>');
            this.elements.startBtn.addClass('stop-btn');
            this.elements.resetBtn.prop('disabled', true);
            this.elements.lapBtn.prop('disabled', false);
            this.elements.timerIcon.removeClass('bi-stopwatch').addClass('bi-stopwatch-fill');
        }
    }

    // Detiene el cronómetro
    stopTimer() {
        if (this.running) {
            this.running = false;
            clearInterval(this.timerInterval);
            this.timerInterval = null;

            // Actualiza la UI
            this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Continuar</span>');
            this.elements.startBtn.removeClass('stop-btn');
            this.elements.resetBtn.prop('disabled', false);
            this.elements.lapBtn.prop('disabled', true);
            this.elements.timerIcon.removeClass('bi-stopwatch-fill').addClass('bi-stopwatch');
        }
    }

    // Reinicia el cronómetro
    resetTimer() {
        this.stopTimer();
        this.elapsedTime = 0;
        this.laps = [];
        
        // Actualiza la UI
        this.updateTimerDisplay();
        this.elements.startBtn.html('<i class="bi bi-play-fill"></i><span>Iniciar</span>');
        this.elements.resetBtn.prop('disabled', true);
        this.elements.lapBtn.prop('disabled', true);
        this.elements.lapsList.empty();
        this.updateLapStats();
    }

    // Actualiza el tiempo transcurrido
    updateElapsedTime() {
        const now = Date.now();
        this.elapsedTime = now - this.startTime;
        this.updateTimerDisplay();
    }

    // Actualiza la pantalla del cronómetro
    updateTimerDisplay() {
        const time = this.formatTime(this.elapsedTime);
        this.elements.timerDisplay.text(time.main);
        this.elements.milliseconds.text(time.ms);
        this.elements.totalTime.text(`Tiempo Total: ${time.main}`);
    }

    // Añade una vuelta
    addLap() {
        if (this.running) {
            const lapTime = this.elapsedTime;
            const previousLap = this.laps.length > 0 ? this.laps[this.laps.length - 1].time : 0;
            const lapDelta = previousLap > 0 ? lapTime - previousLap : 0;
            
            this.laps.push({
                number: this.laps.length + 1,
                time: lapTime,
                lapTime: lapTime - (previousLap || 0),
                delta: lapDelta
            });
            
            this.updateLapsList();
            this.updateLapStats();
        }
    }

    // Actualiza la lista de vueltas
    updateLapsList() {
        // Limpia la lista actual
        this.elements.lapsList.empty();
        
        // Agrega cada vuelta en orden inverso (la más reciente arriba)
        [...this.laps].reverse().forEach(lap => {
            const lapTimeFormatted = this.formatTime(lap.lapTime).main;
            
            // Determina si esta vuelta fue más rápida o más lenta que la anterior
            let deltaClass = '';
            let deltaText = '';
            
            if (lap.number > 1) {
                const avgTime = this.calculateAverageLapTime(lap.number - 1);
                if (lap.lapTime < avgTime) {
                    deltaClass = 'faster';
                    deltaText = `-${this.formatDeltaTime(avgTime - lap.lapTime)}`;
                } else if (lap.lapTime > avgTime) {
                    deltaClass = 'slower';
                    deltaText = `+${this.formatDeltaTime(lap.lapTime - avgTime)}`;
                }
            }
            
            const lapElement = `
                <div class="lap-item">
                    <div class="lap-number">Vuelta ${lap.number}</div>
                    <div class="lap-time">${lapTimeFormatted}</div>
                    ${deltaText ? `<div class="lap-delta ${deltaClass}">${deltaText}</div>` : '<div></div>'}
                </div>
            `;
            
            this.elements.lapsList.append(lapElement);
        });
    }

    // Actualiza las estadísticas de las vueltas
    updateLapStats() {
        const lapCount = this.laps.length;
        this.elements.lapCount.text(`Vueltas: ${lapCount}`);
        
        if (lapCount > 0) {
            const avgTime = this.calculateAverageLapTime();
            const avgTimeFormatted = this.formatTime(avgTime).main;
            this.elements.avgLap.text(`Promedio: ${avgTimeFormatted}`);
        } else {
            this.elements.avgLap.text('Promedio: 00:00:00');
        }
    }
    
    // Calcula el tiempo promedio de vuelta
    calculateAverageLapTime(upToLap = null) {
        if (this.laps.length === 0) return 0;
        
        const lapsToCalculate = upToLap ? this.laps.slice(0, upToLap) : this.laps;
        const totalLapTime = lapsToCalculate.reduce((sum, lap) => sum + lap.lapTime, 0);
        
        return totalLapTime / lapsToCalculate.length;
    }
    
    // Formatea el tiempo en milisegundos a formato HH:MM:SS.ms
    formatTime(timeInMs) {
        const totalSeconds = Math.floor(timeInMs / 1000);
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;
        const milliseconds = Math.floor((timeInMs % 1000) / 10);
        
        const formattedTime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        const formattedMs = milliseconds.toString().padStart(2, '0');
        
        return {
            main: formattedTime,
            ms: formattedMs
        };
    }
    
    // Formatea el tiempo delta (para mostrar diferencias entre vueltas)
    formatDeltaTime(timeInMs) {
        const totalSeconds = Math.floor(timeInMs / 1000);
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = totalSeconds % 60;
        const milliseconds = Math.floor((timeInMs % 1000) / 10);
        
        if (minutes > 0) {
            return `${minutes}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(2, '0')}`;
        }
        
        return `${seconds}.${milliseconds.toString().padStart(2, '0')}`;
    }
}

// Iniciar la aplicación cuando el DOM esté listo
$(document).ready(() => {
    window.timerApp = new TimerApp();
});