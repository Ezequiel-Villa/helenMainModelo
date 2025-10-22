// Variables globales
let selectedIcon = '';
let selectedAddIcon = '';
let deviceToDeleteId = null;
let deviceToEditId = null;
let nextDeviceId = 1;
let selectedRoom = '';
let selectedType = '';

// Inicialización al cargar la página
document.addEventListener('DOMContentLoaded', function() {
  // Comprobar si hay dispositivos
  checkEmptyState();
});

// Funciones para el modal de eliminar
function showDeleteModal(deviceId) {
  deviceToDeleteId = deviceId;
  document.getElementById('deleteModal').style.display = 'flex';
}

function hideDeleteModal() {
  document.getElementById('deleteModal').style.display = 'none';
  deviceToDeleteId = null;
}

function confirmDelete() {
  if (deviceToDeleteId) {
    const deviceElement = document.getElementById(`device-${deviceToDeleteId}`);
    if (deviceElement) {
      deviceElement.remove();
      
      // Verificar si quedan dispositivos
      checkEmptyState();
    }
    hideDeleteModal();
  }
}

// Funciones para el modal de editar
function showEditModal(deviceId) {
  deviceToEditId = deviceId;
  
  // Obtener los datos actuales del dispositivo
  const deviceElement = document.getElementById(`device-${deviceId}`);
  const currentName = deviceElement.querySelector('.device-name').textContent;
  const currentIcon = deviceElement.querySelector('.device-icon').textContent;
  
  // Extraer ubicación y tipo del nombre actual (si sigue el formato establecido)
  const nameParts = currentName.split(' - ');
  if (nameParts.length === 2) {
    selectRoom(document.querySelector(`#editModal .room-option[data-room="${nameParts[0]}"]`), nameParts[0]);
    selectDeviceType(document.querySelector(`#editModal .type-option[data-type="${nameParts[1]}"]`), nameParts[1]);
  }
  
  // Resetear las selecciones de íconos
  const iconOptions = document.querySelectorAll('#editModal .icon-option');
  iconOptions.forEach(option => {
    option.classList.remove('selected');
    if (option.textContent === currentIcon) {
      option.classList.add('selected');
      selectedIcon = currentIcon;
    }
  });
  
  document.getElementById('editModal').style.display = 'flex';
  updateDeviceNamePreview('edit');
}

function hideEditModal() {
  document.getElementById('editModal').style.display = 'none';
  deviceToEditId = null;
  selectedRoom = '';
  selectedType = '';
}

function selectIcon(element, icon) {
  // Quitar la clase selected de todos los íconos
  const iconOptions = document.querySelectorAll('#editModal .icon-option');
  iconOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected al ícono seleccionado
  element.classList.add('selected');
  selectedIcon = icon;
  
  // Sugerir automáticamente un tipo de dispositivo basado en el ícono
  suggestDeviceType(icon, 'edit');
}

function selectRoom(element, room) {
  // Quitar la clase selected de todas las habitaciones
  const roomOptions = document.querySelectorAll('#editModal .room-option');
  roomOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected a la habitación seleccionada
  element.classList.add('selected');
  selectedRoom = room;
  
  // Actualizar el preview del nombre
  updateDeviceNamePreview('edit');
}

function selectDeviceType(element, type) {
  // Quitar la clase selected de todos los tipos
  const typeOptions = document.querySelectorAll('#editModal .type-option');
  typeOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected al tipo seleccionado
  element.classList.add('selected');
  selectedType = type;
  
  // Actualizar el preview del nombre
  updateDeviceNamePreview('edit');
}

function updateDeviceNamePreview(mode) {
  const previewElement = document.getElementById(mode === 'edit' ? 'deviceNamePreview' : 'newDeviceNamePreview');
  
  if (selectedRoom && selectedType) {
    previewElement.textContent = `${selectedRoom} - ${selectedType}`;
    previewElement.parentElement.classList.remove('name-preview-empty');
  } else if (selectedRoom) {
    previewElement.textContent = `${selectedRoom} - ...`;
    previewElement.parentElement.classList.remove('name-preview-empty');
  } else if (selectedType) {
    previewElement.textContent = `... - ${selectedType}`;
    previewElement.parentElement.classList.remove('name-preview-empty');
  } else {
    previewElement.textContent = 'Selecciona ubicación y tipo';
    previewElement.parentElement.classList.add('name-preview-empty');
  }
}

function suggestDeviceType(icon, mode) {
  // Sugerir tipo de dispositivo basado en el ícono
  let suggestedType = '';
  
  switch(icon) {
    case '💡': suggestedType = 'Luz'; break;
    case '🔌': suggestedType = 'Enchufe'; break;
    case '🌡️': suggestedType = 'Termostato'; break;
    case '📺': suggestedType = 'TV'; break;
    case '🔊': suggestedType = 'Altavoz'; break;
    case '🎮': suggestedType = 'Consola'; break;
    case '🚪': suggestedType = 'Puerta'; break;
    case '🔒': suggestedType = 'Cerradura'; break;
    case '💧': suggestedType = 'Riego'; break;
    case '☕': suggestedType = 'Cafetera'; break;
    case '💻': suggestedType = 'Ordenador'; break;
    case '⏰': suggestedType = 'Alarma'; break;
    case '📱': suggestedType = 'Móvil'; break;
    case '🎛️': suggestedType = 'Control'; break;
    case '🖨️': suggestedType = 'Impresora'; break;
    default: return; // No sugerir nada si no hay coincidencia
  }
  
  // Seleccionar el tipo sugerido
  const typeElement = document.querySelector(`#${mode === 'edit' ? 'editModal' : 'addModal'} .type-option[data-type="${suggestedType}"]`);
  if (typeElement) {
    if (mode === 'edit') {
      selectDeviceType(typeElement, suggestedType);
    } else {
      selectAddDeviceType(typeElement, suggestedType);
    }
  }
}

function saveDeviceChanges() {
  if (!selectedRoom || !selectedType) {
    alert('Por favor selecciona una ubicación y un tipo para el dispositivo');
    return;
  }
  
  if (!selectedIcon) {
    alert('Por favor selecciona un ícono para el dispositivo');
    return;
  }
  
  const deviceName = `${selectedRoom} - ${selectedType}`;
  
  if (deviceToEditId) {
    const deviceElement = document.getElementById(`device-${deviceToEditId}`);
    if (deviceElement) {
      // Actualizar para la nueva estructura
      const deviceHeader = deviceElement.querySelector('.device-header');
      if (deviceHeader) {
        // Si ya tiene la nueva estructura
        deviceElement.querySelector('.device-name').textContent = deviceName;
        deviceElement.querySelector('.device-icon').textContent = selectedIcon;
      } else {
        // Si tiene la estructura antigua, actualizar al nuevo formato
        const oldIcon = deviceElement.querySelector('.device-icon');
        const oldName = deviceElement.querySelector('.device-name');
        const status = deviceElement.querySelector('.device-status');
        const actions = deviceElement.querySelector('.device-actions');
        
        // Crear nueva estructura
        const header = document.createElement('div');
        header.className = 'device-header';
        
        // Mover icono y nombre al header
        if (oldIcon && oldName) {
          oldIcon.textContent = selectedIcon;
          oldName.textContent = deviceName;
          header.appendChild(oldIcon);
          header.appendChild(oldName);
          
          // Reestructurar el device-card
          deviceElement.innerHTML = '';
          deviceElement.appendChild(header);
          
          // Añadir status badge si no existe
          const statusText = status.querySelector('.device-status-text');
          const isOn = statusText.textContent === 'Encendido';
          const statusBadge = document.createElement('span');
          statusBadge.className = isOn ? 'status-badge status-on' : 'status-badge status-off';
          statusBadge.textContent = isOn ? 'ON' : 'OFF';
          status.appendChild(statusBadge);
          
          deviceElement.appendChild(status);
          deviceElement.appendChild(actions);
        } else {
          // Simplemente actualizar texto
          deviceElement.querySelector('.device-name').textContent = deviceName;
          deviceElement.querySelector('.device-icon').textContent = selectedIcon;
        }
      }
    }
  }
  
  hideEditModal();
}

// Funciones para el modal de agregar
function showAddModal() {
  selectedAddIcon = '';
  selectedRoom = '';
  selectedType = '';
  
  // Resetear las selecciones de íconos
  const iconOptions = document.querySelectorAll('#addModal .icon-option');
  iconOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Resetear las selecciones de habitaciones y tipos
  const roomOptions = document.querySelectorAll('#addModal .room-option');
  roomOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  const typeOptions = document.querySelectorAll('#addModal .type-option');
  typeOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Restablecer preview
  updateDeviceNamePreview('add');
  
  document.getElementById('addModal').style.display = 'flex';
}

function hideAddModal() {
  document.getElementById('addModal').style.display = 'none';
  selectedRoom = '';
  selectedType = '';
}

function selectAddIcon(element, icon) {
  // Quitar la clase selected de todos los íconos
  const iconOptions = document.querySelectorAll('#addModal .icon-option');
  iconOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected al ícono seleccionado
  element.classList.add('selected');
  selectedAddIcon = icon;
  
  // Sugerir automáticamente un tipo de dispositivo basado en el ícono
  suggestDeviceType(icon, 'add');
}

function selectAddRoom(element, room) {
  // Quitar la clase selected de todas las habitaciones
  const roomOptions = document.querySelectorAll('#addModal .room-option');
  roomOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected a la habitación seleccionada
  element.classList.add('selected');
  selectedRoom = room;
  
  // Actualizar el preview del nombre
  updateDeviceNamePreview('add');
}

function selectAddDeviceType(element, type) {
  // Quitar la clase selected de todos los tipos
  const typeOptions = document.querySelectorAll('#addModal .type-option');
  typeOptions.forEach(option => {
    option.classList.remove('selected');
  });
  
  // Agregar la clase selected al tipo seleccionado
  element.classList.add('selected');
  selectedType = type;
  
  // Actualizar el preview del nombre
  updateDeviceNamePreview('add');
}

function toggleDeviceStatus(checkbox, deviceId) {
  const deviceElement = document.getElementById(`device-${deviceId}`);
  const statusText = deviceElement.querySelector('.device-status-text');
  const statusBadge = deviceElement.querySelector('.status-badge');
  
  if (checkbox.checked) {
    statusText.textContent = 'Encendido';
    if (statusBadge) {
      statusBadge.textContent = 'ON';
      statusBadge.className = 'status-badge status-on';
    }
  } else {
    statusText.textContent = 'Apagado';
    if (statusBadge) {
      statusBadge.textContent = 'OFF';
      statusBadge.className = 'status-badge status-off';
    }
  }
}

function addNewDevice() {
  if (!selectedRoom || !selectedType) {
    alert('Por favor selecciona una ubicación y un tipo para el dispositivo');
    return;
  }
  
  if (!selectedAddIcon) {
    alert('Por favor selecciona un ícono para el dispositivo');
    return;
  }
  
  const deviceName = `${selectedRoom} - ${selectedType}`;
  
  // Remover el mensaje de estado vacío si existe
  const emptyState = document.querySelector('.empty-state');
  if (emptyState) {
    emptyState.remove();
  }
  
  // Crear nuevo dispositivo con la estructura mejorada
  const deviceList = document.getElementById('deviceList');
  const newDevice = document.createElement('div');
  newDevice.className = 'device-card';
  newDevice.id = `device-${nextDeviceId}`;
  
  newDevice.innerHTML = `
    <div class="device-header">
      <div class="device-icon">${selectedAddIcon}</div>
      <div class="device-name">${deviceName}</div>
    </div>
    <div class="device-status">
      <label class="switch">
        <input type="checkbox" onchange="toggleDeviceStatus(this, ${nextDeviceId})">
        <span class="slider"></span>
      </label>
      <span class="device-status-text">Apagado</span>
      <span class="status-badge status-off">OFF</span>
    </div>
    <div class="device-actions">
      <button class="action-button" onclick="showEditModal(${nextDeviceId})">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
      <button class="action-button" onclick="showDeleteModal(${nextDeviceId})">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2M10 11v6M14 11v6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  `;
  
  deviceList.appendChild(newDevice);
  nextDeviceId++;
  
  hideAddModal();
}

// Función para verificar si quedan dispositivos
function checkEmptyState() {
  const deviceList = document.getElementById('deviceList');
  if (deviceList.children.length === 0) {
    const emptyState = document.createElement('div');
    emptyState.className = 'empty-state';
    emptyState.innerHTML = `
      <p>No tienes dispositivos agregados</p>
      <button class="add-button" onclick="showAddModal()">+ Agregar dispositivo</button>
    `;
    deviceList.appendChild(emptyState);
  }
}