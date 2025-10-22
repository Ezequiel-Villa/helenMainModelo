// Inicialización de variables globales
let timer = null;
let startTime = 0;
let elapsedTime = 0;
let totalTime = 0;
let isRunning = false;
let presets = JSON.parse(localStorage.getItem('timerPresets')) || [];
let pausedTime = 0;

// Elementos DOM
const timerDisplay = document.querySelector('.timer-display');
const millisecondsDisplay = document.querySelector('.milliseconds');
const hoursInput = document.getElementById('hours-input');
const minutesInput = document.getElementById('minutes-input');
const secondsInput = document.getElementById('seconds-input');
const startBtn = document.querySelector('.start-btn');
const resetBtn = document.querySelector('.reset-btn');
const presetsContainer = document.querySelector('.presets-list');
const addPresetBtn = document.querySelector('.add-preset-btn');
const timerIcon = document.querySelector('.timer-icon');
const currentTimer = document.querySelector('.current-timer');
const dateSection = document.querySelector('.date-section');

// Lista de nombres de presets predefinidos
const predefinedPresetNames = [
    'Ejercicio',
    'Meditación',
    'Descanso',
    'Cocinar',
    'Trote',
    'Caminata',
    'Carrera',
];


function updateDateTime() {
    const now = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    let dateText = now.toLocaleDateString('es-ES', options);
    
    // Capitalizar el día de la semana (primera palabra)
    dateText = dateText.charAt(0).toUpperCase() + dateText.slice(1);
    
    // Capitalizar el mes (buscar patrón "de nombre mes")
    dateText = dateText.replace(/(de )([a-z]+)/g, function(match, p1, p2) {
        return p1 + p2.charAt(0).toUpperCase() + p2.slice(1);
    });
    
    dateSection.textContent = dateText;
}

// Convertir tiempo total en segundos a formato HH:MM:SS
function formatTime(totalSeconds) {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = Math.floor(totalSeconds % 60);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

// Formatear milisegundos (solo los últimos 2 dígitos)
function formatMilliseconds(ms) {
    return Math.floor((ms % 1000) / 10).toString().padStart(2, '0');
}

// Actualizar la visualización del temporizador
function updateDisplay(timeInMs, isCountingDown = true) {
    if (isCountingDown) {
        const remaining = totalTime - timeInMs;
        if (remaining <= 0) {
            timerDisplay.textContent = '00:00:00';
            millisecondsDisplay.textContent = '00';
            return true; // Tiempo completado
        }

        const formattedTime = formatTime(Math.ceil(remaining / 1000));
        const formattedMs = formatMilliseconds(remaining);
        timerDisplay.textContent = formattedTime;
        millisecondsDisplay.textContent = formattedMs;
    } else {
        const formattedTime = formatTime(Math.floor(timeInMs / 1000));
        const formattedMs = formatMilliseconds(timeInMs);
        timerDisplay.textContent = formattedTime;
        millisecondsDisplay.textContent = formattedMs;
    }
    return false; // Tiempo no completado
}

// Iniciar/Detener el temporizador
function toggleTimer() {
    if (!isRunning) {
        if (pausedTime === 0) {
            // Primera vez que se inicia el temporizador
            // Obtener tiempo de los inputs
            const hours = parseInt(hoursInput.value) || 0;
            const minutes = parseInt(minutesInput.value) || 0;
            const seconds = parseInt(secondsInput.value) || 0;
            
            // Verificar si hay tiempo establecido
            if (hours === 0 && minutes === 0 && seconds === 0) {
                Swal.fire({
                    title: 'Error',
                    text: 'Por favor, establece un tiempo para el temporizador',
                    icon: 'error',
                    confirmButtonText: 'OK'
                });
                return;
            }
            
            // Calcular tiempo total en milisegundos
            totalTime = (hours * 3600 + minutes * 60 + seconds) * 1000;
            
            // Deshabilitar inputs de tiempo
            hoursInput.disabled = true;
            minutesInput.disabled = true;
            secondsInput.disabled = true;
        }
        
        // Cambiar interfaz a modo "corriendo"
        startBtn.innerHTML = '<i class="bi bi-pause-fill"></i><span>Pausar</span>';
        startBtn.classList.add('stop-btn');
        resetBtn.disabled = false;
        
        // Iniciar temporizador
        startTime = Date.now() - pausedTime; // Ajustamos el tiempo de inicio restando el tiempo pausado
        isRunning = true;
        
        timer = setInterval(() => {
            elapsedTime = Date.now() - startTime;
            const timerComplete = updateDisplay(elapsedTime);
            
            if (timerComplete) {
                clearInterval(timer);
                isRunning = false;
                pausedTime = 0; // Resetear el tiempo pausado
                timerComplete();
            }
        }, 10);
    } else {
        // Detener temporizador
        clearInterval(timer);
        isRunning = false;
        
        // Guardar el tiempo transcurrido hasta ahora
        pausedTime = Date.now() - startTime;
        
        // Cambiar interfaz a modo "pausado"
        startBtn.innerHTML = '<i class="bi bi-play-fill"></i><span>Reanudar</span>';
        startBtn.classList.remove('stop-btn');
    }
}

// Función para manejar la finalización del temporizador
function timerComplete() {
    // Cambiar interfaz a modo "completado"
    startBtn.innerHTML = '<i class="bi bi-play-fill"></i><span>Iniciar</span>';
    startBtn.classList.remove('stop-btn');
    
    // Habilitar inputs de tiempo
    hoursInput.disabled = false;
    minutesInput.disabled = false;
    secondsInput.disabled = false;
    
    // Notificación
    Swal.fire({
        title: '¡Tiempo terminado!',
        text: 'El temporizador ha finalizado',
        icon: 'success',
        confirmButtonText: 'OK'
    });
}

// Resetear el temporizador
function resetTimer() {
    clearInterval(timer);
    isRunning = false;
    
    // Resetear interfaz
    timerDisplay.textContent = '00:00:00';
    millisecondsDisplay.textContent = '00';
    
    // Resetear botón de inicio
    startBtn.innerHTML = '<i class="bi bi-play-fill"></i><span>Iniciar</span>';
    startBtn.classList.remove('stop-btn');
    resetBtn.disabled = true;
    
    // Habilitar inputs de tiempo
    hoursInput.disabled = false;
    minutesInput.disabled = false;
    secondsInput.disabled = false;

    pausedTime = 0; // Resetear el tiempo pausado
}

// Aplicar un preset
function applyPreset(preset) {
    hoursInput.value = preset.hours;
    minutesInput.value = preset.minutes;
    secondsInput.value = preset.seconds;
}

// Guardar un nuevo preset
function savePreset() {
    const hours = parseInt(hoursInput.value) || 0;
    const minutes = parseInt(minutesInput.value) || 0;
    const seconds = parseInt(secondsInput.value) || 0;
    
    if (hours === 0 && minutes === 0 && seconds === 0) {
        Swal.fire({
            title: 'Error',
            text: 'Por favor, establece un tiempo para guardar el preset',
            icon: 'error',
            confirmButtonText: 'OK'
        });
        return;
    }
    
    // Crear opciones HTML para el select
    const selectOptions = predefinedPresetNames.map(name => 
        `<option value="${name}">${name}</option>`
    ).join('');
    
    Swal.fire({
        title: 'Guardar Preset',
        html: `
            <label for="preset-select" style="display: block; margin-bottom: 10px; font-weight: bold;">
                Selecciona el nombre del Preset:
            </label>
            <select id="preset-select" class="swal2-input" style="width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 6px;">
                <option value="">-- Selecciona un preset --</option>
                ${selectOptions}
            </select>
        `,
        showCancelButton: true,
        confirmButtonText: 'Guardar',
        cancelButtonText: 'Cancelar',
        preConfirm: () => {
            const selectedName = document.getElementById('preset-select').value;
            if (!selectedName) {
                Swal.showValidationMessage('Por favor selecciona un nombre para el preset');
                return false;
            }
            return selectedName;
        }
    }).then((result) => {
        if (result.isConfirmed) {
            const selectedName = result.value;
            
            // Verificar si ya existe un preset con ese nombre
            const existingPresetIndex = presets.findIndex(preset => preset.name === selectedName);
            
            if (existingPresetIndex !== -1) {
                // Si existe, preguntar si quiere sobreescribir
                Swal.fire({
                    title: 'Preset existente',
                    text: `Ya existe un preset llamado "${selectedName}". ¿Quieres sobreescribirlo?`,
                    icon: 'warning',
                    showCancelButton: true,
                    confirmButtonText: 'Sí, sobreescribir',
                    cancelButtonText: 'Cancelar'
                }).then((overwriteResult) => {
                    if (overwriteResult.isConfirmed) {
                        // Sobreescribir el preset existente
                        presets[existingPresetIndex] = {
                            id: presets[existingPresetIndex].id,
                            name: selectedName,
                            hours: hours,
                            minutes: minutes,
                            seconds: seconds
                        };
                        
                        localStorage.setItem('timerPresets', JSON.stringify(presets));
                        renderPresets();
                        
                        Swal.fire({
                            title: '¡Actualizado!',
                            text: 'Preset actualizado exitosamente',
                            icon: 'success',
                            confirmButtonText: 'OK'
                        });
                    }
                });
            } else {
                // Crear nuevo preset
                const newPreset = {
                    id: Date.now(),
                    name: selectedName,
                    hours: hours,
                    minutes: minutes,
                    seconds: seconds
                };
                
                presets.push(newPreset);
                localStorage.setItem('timerPresets', JSON.stringify(presets));
                renderPresets();
                
                Swal.fire({
                    title: '¡Guardado!',
                    text: 'Preset guardado exitosamente',
                    icon: 'success',
                    confirmButtonText: 'OK'
                });
            }
        }
    });
}

// Eliminar un preset
function deletePreset(id) {
    Swal.fire({
        title: '¿Estás seguro?',
        text: 'Esta acción no se puede deshacer',
        icon: 'warning',
        showCancelButton: true,
        confirmButtonText: 'Sí, eliminar',
        cancelButtonText: 'Cancelar'
    }).then((result) => {
        if (result.isConfirmed) {
            presets = presets.filter(preset => preset.id !== id);
            localStorage.setItem('timerPresets', JSON.stringify(presets));
            renderPresets();
            
            Swal.fire({
                title: 'Eliminado',
                text: 'El preset ha sido eliminado',
                icon: 'success',
                confirmButtonText: 'OK'
            });
        }
    });
}

// Renderizar lista de presets
function renderPresets() {
    presetsContainer.innerHTML = '';
    
    if (presets.length === 0) {
        presetsContainer.innerHTML = '<p style="text-align: center; padding: 20px;">No hay presets guardados</p>';
        return;
    }
    
    presets.forEach(preset => {
        const formattedTime = formatTime(preset.hours * 3600 + preset.minutes * 60 + preset.seconds);
        
        const presetElement = document.createElement('div');
        presetElement.classList.add('preset-item');
        presetElement.innerHTML = `
            <div class="preset-info">
                <span class="preset-name">${preset.name}</span>
                <span class="preset-time">${formattedTime}</span>
            </div>
            <div class="preset-actions">
                <button class="preset-btn preset-use" data-id="${preset.id}">
                    <i class="bi bi-arrow-right-circle"></i>
                </button>
                <button class="preset-btn preset-delete" data-id="${preset.id}">
                    <i class="bi bi-trash3"></i>
                </button>
            </div>
        `;
        
        presetsContainer.appendChild(presetElement);
    });
    
    // Agregar event listeners a botones de presets
    document.querySelectorAll('.preset-use').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = parseInt(btn.dataset.id);
            const preset = presets.find(p => p.id === id);
            if (preset) {
                applyPreset(preset);
            }
        });
    });
    
    document.querySelectorAll('.preset-delete').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = parseInt(btn.dataset.id);
            deletePreset(id);
        });
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Inicializar fecha y hora
    updateDateTime();
    setInterval(updateDateTime, 60000); // Actualizar cada minuto
    
    // Inicializar lista de presets
    renderPresets();
    
    // Event listeners para botones de control
    startBtn.addEventListener('click', toggleTimer);
    resetBtn.addEventListener('click', resetTimer);
    addPresetBtn.addEventListener('click', savePreset);
    
    // Event listeners para inputs de tiempo
    [hoursInput, minutesInput, secondsInput].forEach(input => {
        input.addEventListener('change', () => {
            // Validar valores
            if (parseInt(input.value) < 0) {
                input.value = 0;
            } else if (input.id === 'hours-input' && parseInt(input.value) > 23) {
                input.value = 23;
            } else if ((input.id === 'minutes-input' || input.id === 'seconds-input') && parseInt(input.value) > 59) {
                input.value = 59;
            }
        });
    });
    
    // Control de botones flecha para inputs de tiempo
    document.querySelectorAll('.up-arrow').forEach(btn => {
        btn.addEventListener('click', () => {
            const input = btn.parentElement.querySelector('input');
            let value = parseInt(input.value) || 0;
            
            if (input.id === 'hours-input' && value < 23) {
                input.value = value + 1;
            } else if ((input.id === 'minutes-input' || input.id === 'seconds-input') && value < 59) {
                input.value = value + 1;
            }
        });
    });
    
    document.querySelectorAll('.down-arrow').forEach(btn => {
        btn.addEventListener('click', () => {
            const input = btn.parentElement.querySelector('input');
            let value = parseInt(input.value) || 0;
            
            if (value > 0) {
                input.value = value - 1;
            }
        });
    });
});