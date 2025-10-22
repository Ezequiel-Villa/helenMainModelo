let selectedNetwork = null;
let currentConnection = null;
let networks = [];

const wifiButton = document.getElementById('wifi-button');
const wifiStatusIcon = document.getElementById('wifi-status-icon');
const wifiModal = document.getElementById('wifi-modal');
const closeWifiModal = document.getElementById('close-wifi-modal');
const wifiList = document.getElementById('wifi-list');
const wifiStatus = document.getElementById('wifi-status');
const connectedNetworkText = document.getElementById('connected-network');
const refreshWifiButton = document.getElementById('refresh-wifi');
const connectWifiButton = document.getElementById('connect-wifi-button');
const passwordContainer = document.getElementById('password-container');
const passwordInput = document.getElementById('wifi-password');
const togglePassword = document.getElementById('toggle-password');
const passwordToggleIcon = document.getElementById('password-toggle-icon');
const connectLoader = document.getElementById('connect-loader');
const connectText = document.getElementById('connect-text');
const connectingText = document.getElementById('connecting-text');

togglePassword.addEventListener('click', () => {
    const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
    passwordInput.setAttribute('type', type);
    passwordToggleIcon.className = type === 'password' ? 'bi bi-eye-slash' : 'bi bi-eye';
});

function updateWifiIcon(isConnected) {
    wifiStatusIcon.className = isConnected ? 'bi bi-wifi wifi-icon' : 'bi bi-wifi-off wifi-icon';
}

function updateConnectionStatus() {
    if (currentConnection) {
        // connectedNetworkText.textContent = `Conectado a: ${currentConnection.ssid}`;
        // wifiStatus.style.color = 'var(--primary-light)';
    } else {
        // connectedNetworkText.textContent = 'No conectado';
        // wifiStatus.style.color = 'var(--text-light)';
    }
}

async function checkCurrentConnection() {
    try {
        const currentNetworks = await window.wifiAPI.getCurrentConnections(); 
        if (currentNetworks && currentNetworks.length > 0) {
            currentConnection = currentNetworks[0];
            updateWifiIcon(true);
            // updateConnectionStatus();
            localStorage.setItem('wifiConnected', 'true');
            localStorage.setItem('currentSSID', currentConnection.ssid);
            return true;
        } else {
            currentConnection = null;
            updateWifiIcon(false);
            // updateConnectionStatus();
            localStorage.setItem('wifiConnected', 'false');
            localStorage.removeItem('currentSSID');
            return false;
        }
    } catch (error) {
        console.error('Error al verificar la conexión:', error);
        updateWifiIcon(false);
        // updateConnectionStatus();
        return false;
    }
}

async function scanNetworks() {
    try {
        selectedNetwork = null;
        passwordContainer.style.display = 'none';
        passwordInput.value = '';
        connectWifiButton.disabled = true;

        wifiList.innerHTML = '<div class="wifi-item"><div class="wifi-signal"><i class="bi bi-arrow-repeat"></i></div><div class="wifi-name">Buscando redes...</div></div>';

        networks = await window.wifiAPI.scanNetworks();
        networks.sort((a, b) => b.quality - a.quality);
        wifiList.innerHTML = '';

        if (networks.length === 0) {
            wifiList.innerHTML = '<div class="wifi-item"><div class="wifi-signal"><i class="bi bi-exclamation-circle"></i></div><div class="wifi-name">No se encontraron redes</div></div>';
            return;
        }

        if (currentConnection) {
            networks.sort((a, b) => (a.ssid === currentConnection.ssid ? -1 : 1));
        }

        networks.forEach((network, index) => {
            const networkItem = document.createElement('div');
            networkItem.className = 'wifi-item';
            networkItem.dataset.index = index;

            let signalIcon = 'bi-wifi-1';
            if (network.quality >= 70) signalIcon = 'bi-wifi';
            else if (network.quality >= 40) signalIcon = 'bi-wifi-2';

            const isCurrentNetwork = currentConnection && network.ssid === currentConnection.ssid;
            if (isCurrentNetwork) networkItem.classList.add('connected-network');

            networkItem.innerHTML = `
                <div class="wifi-signal"><i class="bi ${signalIcon}"></i></div>
                <div class="wifi-name">${network.ssid} ${isCurrentNetwork ? '<span class="connected-label">(Conectado)</span>' : ''}</div>
                <div class="wifi-secured">${network.security !== '' ? '<i class="bi bi-lock"></i>' : ''}</div>
            `;

            networkItem.addEventListener('click', () => {
                document.querySelectorAll('.wifi-item.selected').forEach(item => item.classList.remove('selected'));
                networkItem.classList.add('selected');
                selectedNetwork = network;
                connectWifiButton.disabled = false;
                passwordContainer.style.display = network.security !== '' ? 'block' : 'none';
                connectText.textContent = isCurrentNetwork ? 'Reconectar' : 'Conectar';
            });

            wifiList.appendChild(networkItem);
        });
    } catch (error) {
        console.error('Error al escanear redes:', error);
        wifiList.innerHTML = '<div class="wifi-item"><div class="wifi-signal"><i class="bi bi-exclamation-triangle"></i></div><div class="wifi-name">Error al buscar redes</div></div>';
    }
}

async function connectToNetwork() {
    if (!selectedNetwork) return;

    connectLoader.style.display = 'inline-block';
    connectText.style.display = 'none';
    connectingText.style.display = 'inline';
    connectWifiButton.disabled = true;

    const password = passwordInput.value;

    try {
        const current = await window.wifiAPI.getCurrentConnections();
        const isChangingNetwork = current && current.length > 0 && current[0].ssid !== selectedNetwork.ssid;

        // Si está conectado a otra red diferente, olvidarla antes de conectar
        if (isChangingNetwork) {
            const currentSSID = current[0].ssid;
            await window.wifiAPI.forget(currentSSID);
        }

        await window.wifiAPI.connect({
            ssid: selectedNetwork.ssid,
            password: selectedNetwork.security ? password : undefined
        });

        await new Promise(resolve => setTimeout(resolve, 5000));
        const connected = await checkCurrentConnection();
        
        if (connected) {
            Swal.fire({
                icon: 'success',
                title: 'Conectado',
                text: `Conectado exitosamente a ${selectedNetwork.ssid}`,
                timer: 3000,
                showConfirmButton: false,
                background: 'var(--card-bg)',
                color: 'var(--text)'
            });
            
            setTimeout(() => { wifiModal.style.display = 'none'; }, 1500);
            setTimeout(() => { location.reload(); }, 2000);
        } else {
            throw new Error('No se pudo establecer conexión.');
        }
    } catch (error) {
        console.error('Error al conectar:', error);

        // Esperar a que el sistema tenga tiempo de liberar la red anterior
        await new Promise(resolve => setTimeout(resolve, 1500));

        const stillConnected = await checkCurrentConnection();

        // Si no estamos conectados, actualizar ícono y estado manualmente (por si acaso)
        if (!stillConnected) {
            updateWifiIcon(false);
            currentConnection = null;
            // updateConnectionStatus();
        }

        Swal.fire({
            icon: 'error',
            title: 'Error de conexión',
            text: `No se pudo conectar a ${selectedNetwork.ssid}. Verifica la contraseña e intenta de nuevo.`,
            confirmButtonText: 'OK',
            background: 'var(--card-bg)',
            color: 'var(--text)',
            confirmButtonColor: 'var(--primary)',
            borderRadius: '15px',
            scrollbarPadding: false
        });
    } finally {
        connectLoader.style.display = 'none';
        connectText.style.display = 'inline';
        connectingText.style.display = 'none';
        connectWifiButton.disabled = false;
    }
}

window.electronAPI.onWifiStatusChange((event, data) => {
    updateWifiIcon(data.isConnected);
    currentConnection = data.currentConnection;
    // updateConnectionStatus();
});

document.addEventListener('DOMContentLoaded', () => {
    const wifiConnected = localStorage.getItem('wifiConnected');
    updateWifiIcon(wifiConnected === 'true');
    checkCurrentConnection();
});

wifiButton.addEventListener('click', () => {
    wifiModal.style.display = 'flex';
    checkCurrentConnection();
    scanNetworks();
});

closeWifiModal.addEventListener('click', () => {
    wifiModal.style.display = 'none';
});

wifiModal.addEventListener('click', (event) => {
    if (event.target === wifiModal) wifiModal.style.display = 'none';
});

function openModal() {
    wifiModal.style.display = 'flex';
    setTimeout(() => {
        wifiModal.style.opacity = '1';
        document.querySelector('.wifi-modal-content').style.transform = 'scale(1)';
    }, 10);
}

function closeModal() {
    wifiModal.style.opacity = '0';
    document.querySelector('.wifi-modal-content').style.transform = 'scale(0.9)';
    setTimeout(() => { wifiModal.style.display = 'none'; }, 300);
}

refreshWifiButton.addEventListener('click', scanNetworks);
connectWifiButton.addEventListener('click', connectToNetwork);
passwordInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') connectToNetwork();
});

async function toggleDeviceStatus(checkbox) {
    const statusText = document.getElementById('status-text-switch');
    if (checkbox.checked) {
        statusText.textContent = "Encendido";
        refreshWifiButton.hidden = false;
        wifiList.hidden = false;
        connectWifiButton.hidden = false;
        await window.wifiAPI.enableWifi().then(scanNetworks);
    } else {
        statusText.textContent = "Apagado";
        refreshWifiButton.hidden = true;
        wifiList.hidden = true;
        connectWifiButton.hidden = true;
        await window.wifiAPI.disableWifi();
    }
}
