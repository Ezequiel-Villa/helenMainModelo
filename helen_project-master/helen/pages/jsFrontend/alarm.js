document.addEventListener('DOMContentLoaded', function() {
    // Mover el reloj digital junto al logo
    const logo = document.querySelector('.logo');
    const currentTime = document.querySelector('.current-time');
    
    // Remover el reloj de su posición original
    currentTime.remove();
    
    // Modificar estilos para colocarlo junto al logo
    currentTime.style.margin = '0 0 0 15px';
    currentTime.style.padding = '10px 15px';
    currentTime.style.maxWidth = '200px';
    
    // Modificar la estructura interna
    const time = currentTime.querySelector('.time');
    time.style.fontSize = '28px';
    time.style.marginBottom = '2px';
    
    // Insertar junto al logo
    logo.parentNode.appendChild(currentTime);
    
    // Reorganizar el header para que se vea bien
    const header = document.querySelector('.header');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    
    // Ajustar el contenedor principal para el nuevo diseño
    const alarmContainer = document.querySelector('.alarm-container');
    alarmContainer.style.paddingTop = '0';

    // Initialize the app
    updateClock();
    setInterval(updateClock, 1000);
    loadAlarms();
    
    // Event Listeners
    document.getElementById('add-alarm').addEventListener('click', addAlarm);
    document.querySelectorAll('.day-selector').forEach(day => {
        day.addEventListener('click', function() {
            this.classList.toggle('active');
        });
    });
    
    // Fix number inputs to be two digits
    const hourInput = document.getElementById('hours');
    const minuteInput = document.getElementById('minutes');
    
    hourInput.addEventListener('input', function() {
        if (this.value > 23) this.value = 23;
        if (this.value < 0) this.value = 0;
    });
    
    minuteInput.addEventListener('input', function() {
        if (this.value > 59) this.value = 59;
        if (this.value < 0) this.value = 0;
    });
});

// Update the digital clock
function updateClock() {
    const now = new Date();
    const timeElement = document.querySelector('.time');
    const dateElement = document.querySelector('.date-section');
    
    // Format time
    let hours = now.getHours();
    const minutes = now.getMinutes().toString().padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    
    hours = hours % 12;
    hours = hours ? hours : 12; // Convert 0 to 12
    hours = hours.toString().padStart(2, '0');
    
    timeElement.textContent = `${hours}:${minutes}`;
    
    // Format date - simplify date format to save space
    const options = { weekday: 'short', day: 'numeric', month: 'short' };
    const formattedDate = now.toLocaleDateString('es-ES', options);
    dateElement.textContent = formattedDate;
    
    // Check if any alarms should ring
    checkAlarms(now);
}

// Store alarms in localStorage
function saveAlarms(alarms) {
    localStorage.setItem('alarms', JSON.stringify(alarms));
}

// Load alarms from localStorage
function loadAlarms() {
    const alarmList = document.querySelector('.alarm-list');
    const loadingElement = document.querySelector('.alarm-loading');
    const noAlarmsMessage = document.querySelector('.no-alarms-message');
    
    // Simulate loading
    setTimeout(() => {
        loadingElement.style.display = 'none';
        
        const alarms = JSON.parse(localStorage.getItem('alarms')) || [];
        
        if (alarms.length === 0) {
            noAlarmsMessage.style.display = 'flex';
        } else {
            noAlarmsMessage.style.display = 'none';
            
            // Clear existing alarms (except loading and no-alarms-message)
            const existingAlarms = document.querySelectorAll('.alarm-item');
            existingAlarms.forEach(alarm => alarm.remove());
            
            // Create alarm elements
            alarms.forEach(alarm => {
                const alarmElement = createAlarmElement(alarm);
                alarmList.appendChild(alarmElement);
            });
        }
    }, 1000); // Simulated loading time
}

// Create a new alarm element
function createAlarmElement(alarm) {
    const template = document.getElementById('alarm-template');
    const alarmElement = template.content.cloneNode(true);
    
    const alarmItem = alarmElement.querySelector('.alarm-item');
    alarmItem.dataset.id = alarm.id;
    
    const timeElement = alarmElement.querySelector('.alarm-time');
    timeElement.textContent = `${alarm.hours}:${alarm.minutes} ${alarm.ampm}`;
    
    const nameElement = alarmElement.querySelector('.alarm-name');
    nameElement.textContent = alarm.name || 'Alarma';
    
    const daysElement = alarmElement.querySelector('.alarm-days');
    const dayNames = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];
    
    if (alarm.days.length === 7) {
        daysElement.textContent = 'Todos los días';
    } else if (alarm.days.length === 0) {
        daysElement.textContent = 'Una vez';
    } else {
        const daysList = alarm.days.map(day => dayNames[day]).join(', ');
        daysElement.textContent = daysList;
    }
    
    const toggleInput = alarmElement.querySelector('.switch input');
    toggleInput.checked = alarm.active;
    toggleInput.addEventListener('change', function() {
        toggleAlarm(alarm.id, this.checked);
    });
    
    const deleteButton = alarmElement.querySelector('.delete-alarm');
    deleteButton.addEventListener('click', function() {
        deleteAlarm(alarm.id);
    });
    
    return alarmElement;
}

// Add a new alarm
function addAlarm() {
    const hours = document.getElementById('hours').value.padStart(2, '0');
    const minutes = document.getElementById('minutes').value.padStart(2, '0');
    const ampm = document.getElementById('alarm-ampm').value;
    const name = document.getElementById('alarm-name').value;
    
    const activeDays = [];
    document.querySelectorAll('.day-selector.active').forEach(day => {
        activeDays.push(parseInt(day.dataset.day));
    });
    
    if (hours === '' || minutes === '') {
        showAlert('error', 'Por favor ingresa una hora válida');
        return;
    }
    
    const alarmId = Date.now().toString();
    const newAlarm = {
        id: alarmId,
        hours: hours,
        minutes: minutes,
        ampm: ampm,
        days: activeDays,
        name: name,
        active: true
    };
    
    // Save to localStorage
    const alarms = JSON.parse(localStorage.getItem('alarms')) || [];
    alarms.push(newAlarm);
    saveAlarms(alarms);
    
    // Reset form
    document.getElementById('alarm-name').value = '';
    document.querySelectorAll('.day-selector.active').forEach(day => {
        day.classList.remove('active');
    });
    
    // Show success message
    showAlert('success', 'Alarma agregada con éxito');
    
    // Reload alarms
    loadAlarms();
}

// Toggle alarm active state
function toggleAlarm(alarmId, active) {
    const alarms = JSON.parse(localStorage.getItem('alarms')) || [];
    const alarmIndex = alarms.findIndex(alarm => alarm.id === alarmId);
    
    if (alarmIndex !== -1) {
        alarms[alarmIndex].active = active;
        saveAlarms(alarms);
    }
}

// Delete an alarm
function deleteAlarm(alarmId) {
    Swal.fire({
        title: '¿Estás seguro?',
        text: "Esta acción no se puede deshacer",
        icon: 'warning',
        showCancelButton: true,
        confirmButtonColor: '#6C5CE7',
        cancelButtonColor: '#d33',
        confirmButtonText: 'Sí, eliminar',
        cancelButtonText: 'Cancelar'
    }).then((result) => {
        if (result.isConfirmed) {
            const alarms = JSON.parse(localStorage.getItem('alarms')) || [];
            const filteredAlarms = alarms.filter(alarm => alarm.id !== alarmId);
            saveAlarms(filteredAlarms);
            
            // Remove the element
            const alarmElement = document.querySelector(`.alarm-item[data-id="${alarmId}"]`);
            if (alarmElement) {
                alarmElement.remove();
            }
            
            // Show no alarms message if needed
            if (filteredAlarms.length === 0) {
                document.querySelector('.no-alarms-message').style.display = 'flex';
            }
            
            showAlert('success', 'Alarma eliminada con éxito');
        }
    });
}

// Check if any alarms should ring
function checkAlarms(currentTime) {
    const alarms = JSON.parse(localStorage.getItem('alarms')) || [];
    
    alarms.forEach(alarm => {
        if (!alarm.active) return;
        
        let hours = parseInt(alarm.hours);
        const minutes = parseInt(alarm.minutes);
        
        // Convert to 24-hour format
        if (alarm.ampm === 'PM' && hours < 12) {
            hours += 12;
        } else if (alarm.ampm === 'AM' && hours === 12) {
            hours = 0;
        }
        
        // Check if alarm should ring
        const now = currentTime;
        const alarmDay = now.getDay(); // 0 = Sunday, 1 = Monday, etc.
        
        if (
            (alarm.days.length === 0 || alarm.days.includes(alarmDay)) && 
            now.getHours() === hours && 
            now.getMinutes() === minutes && 
            now.getSeconds() === 0
        ) {
            ringAlarm(alarm);
        }
    });
}

// Ring the alarm
function ringAlarm(alarm) {
    // Create audio element
    const audio = new Audio('https://cdn.freesound.org/previews/219/219244_4082826-lq.mp3');
    audio.loop = true;
    
    Swal.fire({
        title: alarm.name || 'Alarma',
        html: `<div style="font-size: 24px;">${alarm.hours}:${alarm.minutes} ${alarm.ampm}</div>`,
        icon: 'info',
        confirmButtonText: 'Detener',
        allowOutsideClick: false,
        didOpen: () => {
            audio.play();
        },
        willClose: () => {
            audio.pause();
        }
    });
}

// Show alert message
function showAlert(type, message) {
    Swal.fire({
        toast: true,
        icon: type,
        title: message,
        position: 'top-end',
        showConfirmButton: false,
        timer: 3000,
        timerProgressBar: true
    });
}