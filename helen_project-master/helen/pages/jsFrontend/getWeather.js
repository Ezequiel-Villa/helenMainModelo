// Configuración global
const CONFIG = {
    API_KEY: '6266f75957014a7de4ae0ded34d1e7cc', // API Key de OpenWeatherMap
    STORAGE_KEY: 'selectedCity',
    DEFAULT_CITY: 'tijuana',
    UPDATE_INTERVAL: 60000,
    WEATHER_UPDATE_INTERVAL: 300000, // 5 minutos en milisegundos
    WEATHER_UNITS: 'metric',
    WEATHER_LANG: 'es'
};

// Coordenadas predefinidas para cada municipio de Baja California
const BC_MUNICIPIOS = {
    mexicali: { lat: 32.6633, lon: -115.4678, name: 'Mexicali' },
    tijuana: { lat: 32.5027, lon: -117.00371, name: 'Tijuana' },
    ensenada: { lat: 31.8667, lon: -116.5967, name: 'Ensenada' },
    tecate: { lat: 32.5667, lon: -116.6333, name: 'Tecate' },
    rosarito: { lat: 32.3614, lon: -117.0553, name: 'Playas de Rosarito' },
    chiapas: { lat: 16.7519, lon: -93.1167, name: 'Tuxtla Gutiérrez' },
};

// Mapa de iconos de OpenWeatherMap a Bootstrap Icons
const WEATHER_ICONS = {
    '01d': 'bi-brightness-high',
    '01n': 'bi-moon',
    '02d': 'bi-cloud-sun',
    '02n': 'bi-cloud-moon',
    '03d': 'bi-cloud',
    '03n': 'bi-cloud',
    '04d': 'bi-clouds',
    '04n': 'bi-clouds',
    '09d': 'bi-cloud-drizzle',
    '09n': 'bi-cloud-drizzle',
    '10d': 'bi-cloud-rain',
    '10n': 'bi-cloud-rain',
    '11d': 'bi-cloud-lightning',
    '11n': 'bi-cloud-lightning',
    '13d': 'bi-cloud-snow',
    '13n': 'bi-cloud-snow',
    '50d': 'bi-cloud-haze',
    '50n': 'bi-cloud-haze'
};

// Arreglos para formato de fecha
const DIAS = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];
const DIAS_CORTO = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];
const MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];

// Clase principal para la aplicación del clima
class WeatherApp {
    constructor() {
        // Elementos DOM que se necesitarán
        this.elements = {
            dateSection: $('.date-section'),
            locationSection: $('.location'),
            weatherIcon: $('.weather-icon'),
            temperature: $('.temperature'),
            weatherDescription: $('.status'),
            feelsLike: $('.feels-like'),
            humidity: $('.humidity'),
            windSpeed: $('.wind-speed'),
            forecastContainer: $('.forecast-container'),
            hourlyForecastContainer: null,
            refreshButton: $('.refresh-weather'),
            loadingIndicator: $('.weather-loading'),
            citySelector: null
        };

        // no hay ubicación actual de momento
        this.currentLocation = null;
        
        // Nuevo: intervalo para actualización automática del clima
        this.weatherUpdateInterval = null;

        // Inicializa la aplicación para que todo funcione
        this.init();
    }

    // Llama a las funciones principales
    init() {
        this.createCitySelector();
        this.loadSavedCity();
        this.setupEventListeners();
        this.startDateTimer();
        this.updateWeatherForLocation(this.currentLocation);
        
        // Nuevo: iniciar temporizador para actualización automática del clima
        this.startWeatherUpdateTimer();
    }

    // Nuevo: método para iniciar el temporizador de actualización de clima
    startWeatherUpdateTimer() {
        // Limpiar cualquier intervalo existente primero
        if (this.weatherUpdateInterval) {
            clearInterval(this.weatherUpdateInterval);
        }
        
        // Establecer un nuevo intervalo (5 minutos)
        this.weatherUpdateInterval = setInterval(() => {
            this.debug("Actualizando datos del clima automáticamente (cada 5 minutos)");
            this.updateWeatherForLocation(this.currentLocation);
        }, CONFIG.WEATHER_UPDATE_INTERVAL);
    }

    // Crear el selector de ciudades
    createCitySelector() {
        $('.weather-container').prepend(`
            <div class="city-selector">
                <div class="custom-select-container">
                    <select id="citySelector" class="custom-select">
                        <option value="" selected>Selecciona un municipio</option>
                        <option value="tijuana">Tijuana</option>
                        <option value="mexicali">Mexicali</option>
                        <option value="ensenada">Ensenada</option>
                        <option value="tecate">Tecate</option>
                        <option value="rosarito">Playas de Rosarito</option>
                        <option value="chiapas">Chiapas</option>
                    </select>
                    <div class="select-icon">
                        <i class="bi bi-geo-alt-fill"></i>
                    </div>
                </div>
            </div>
            <div class="auto-update-info">
            </div>
        `);

        // Selecciona al select y lo guarda en el atributo citySelector de elements
        this.elements.citySelector = $('#citySelector');
    }

    // Carga la ciudad guardada o usar la predeterminada
    loadSavedCity() {
        const savedCity = localStorage.getItem(CONFIG.STORAGE_KEY) || CONFIG.DEFAULT_CITY;
        if (BC_MUNICIPIOS[savedCity]) {
            this.currentLocation = BC_MUNICIPIOS[savedCity];
            this.elements.citySelector.val(savedCity);
        } else {
            console.error("Ciudad no encontrada en BC_MUNICIPIOS:", savedCity);
            this.currentLocation = BC_MUNICIPIOS[CONFIG.DEFAULT_CITY];
            this.elements.citySelector.val(CONFIG.DEFAULT_CITY);
        }
    }

    // Event listener para el selector de ciudades
    setupEventListeners() {
        this.elements.citySelector.on('change', (e) => {
            const selectedCity = $(e.target).val();
            if (selectedCity && BC_MUNICIPIOS[selectedCity]) {
                this.currentLocation = BC_MUNICIPIOS[selectedCity];
                localStorage.setItem(CONFIG.STORAGE_KEY, selectedCity);
                this.updateWeatherForLocation(this.currentLocation);
                
                // Reiniciar el temporizador al cambiar de ciudad
                this.startWeatherUpdateTimer();
            }
        });

        // Event listener para el botón de actualizar
        this.elements.refreshButton.on('click', () => {
            this.updateWeatherForLocation(this.currentLocation);
            
            // Reiniciar también el temporizador al actualizar manualmente
            this.startWeatherUpdateTimer();
        });
    }

    startDateTimer() {
        // Inicia el timer para actualizar la fecha
        this.updateDate();
        setInterval(() => this.updateDate(), CONFIG.UPDATE_INTERVAL);
    }

    // Actualiza la fecha mostrada
    updateDate() {
        const date = new Date();
        const dateString = `${DIAS[date.getDay()]}, ${date.getDate()} de ${MESES[date.getMonth()]} de ${date.getFullYear()}`;
        this.elements.dateSection.html(dateString);
    }

    // Actualiza el clima 
    updateWeatherForLocation(locationData) {
        this.elements.loadingIndicator.fadeIn();
        this.getWeather(locationData.lat, locationData.lon);
        this.getTodayHourlyForecast(locationData.lat, locationData.lon);
    }

    async getWeather(lat, lon) {
        try {
            this.debug('Iniciando getWeather');
            const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
            const response = await axios.get(url);
            const data = response.data;

            // Valida datos recibidos
            if (!data || !data.weather || !data.main || !data.wind) {
                throw new Error("Datos de clima no válidos");
            }

            // Obtiene nombre de la ciudad si no está disponible
            let locationName = data.name;
            let country = data.sys.country;

            if (!locationName || locationName === '') {
                const locationData = await this.getReverseGeocoding(lat, lon);
                locationName = locationData.name;
                country = locationData.country;
            }

            const weatherData = {
                icon: data.weather[0].icon,
                temperature: data.main.temp.toFixed(1),
                feelsLike: data.main.feels_like.toFixed(1),
                humidity: data.main.humidity,
                windSpeed: (data.wind.speed * 3.6).toFixed(1), // Convierte m/s to km/h
                description: data.weather[0].description,
                city: locationName,
                country: country
            };

            this.updateWeatherUI(weatherData);

            // Obtiene el pronóstico después de actualizar el clima actual
            await this.getForecast(lat, lon);
            this.debug('getWeather completado con éxito');

        } catch (error) {
            console.error("Error getting weather data:", error);
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'No se pudo obtener la información del clima. Por favor, intenta nuevamente más tarde.'
            });
            this.elements.loadingIndicator.fadeOut();
            throw error; // Propaga el error para que se pueda manejar en updateWeatherForLocation
        }
    }

    // Obtiene ubicacion a travez de coordenadas
    async getReverseGeocoding(lat, lon) {
        try {
            const url = `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${CONFIG.API_KEY}`;
            const response = await axios.get(url);

            if (response.data && response.data.length > 0) {
                return {
                    name: response.data[0].name,
                    country: response.data[0].country
                };
            }
            return { name: 'Desconocido', country: 'Desconocido' };
        } catch (error) {
            console.error("Error en geocodificación inversa:", error);
            return { name: 'Desconocido', country: 'Desconocido' };
        }
    }

    async getForecast(lat, lon) {
        try {
            const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
            const response = await axios.get(url);
            const data = response.data;
            
            // Procesa datos de pronóstico (una entrada por día)
            const forecastData = this.processForecastData(data.list);
            
            // Actualiza la UI con los datos del pronóstico
            this.updateForecastUI(forecastData);
            
            // Procesa y muestra el pronóstico horario
            this.processHourlyForecastData(data.list);
            this.updateHourlyForecastUI(this.processHourlyForecastData(data.list));
            
        } catch (error) {
            console.error("Error getting forecast data:", error);
        }
    }
    
    processForecastData(forecastList) {
        const forecastData = [];
        const processedDates = new Set();
    
        // Primero intenta obtener pronósticos al mediodía
        forecastList.forEach(item => {
            const date = new Date(item.dt * 1000);
            const dateStr = date.toISOString().split('T')[0];
    
            // Solo toma un pronóstico por día (mediodía si es posible)
            if (!processedDates.has(dateStr) && date.getHours() >= 11 && date.getHours() <= 20) {
                processedDates.add(dateStr);
                forecastData.push({
                    date: date,
                    icon: item.weather[0].icon,
                    temp: item.main.temp.toFixed(0),
                    description: item.weather[0].description
                });
            }
        });
    
        // Asegura que tenemos al menos una entrada por día aunque no sea al mediodía
        forecastList.forEach(item => {
            const date = new Date(item.dt * 1000);
            const dateStr = date.toISOString().split('T')[0];
    
            if (!processedDates.has(dateStr)) {
                processedDates.add(dateStr);
                forecastData.push({
                    date: date,
                    icon: item.weather[0].icon,
                    temp: item.main.temp.toFixed(0),
                    description: item.weather[0].description
                });
            }
        });
    
        // Ordena por fecha y limita a 3 días
        return forecastData.sort((a, b) => a.date - b.date).slice(1, 4);
    }

    // Actualiza al informacion visual
    updateWeatherUI(data) {
        // Limpia clases de icono anteriores
        this.elements.weatherIcon.removeClass();
        this.elements.weatherIcon.addClass('weather-icon bi'); // Añade clases base

        // Aplica icono correspondiente
        this.elements.weatherIcon.addClass(WEATHER_ICONS[data.icon] || 'bi-question-circle');

        // Actualiza elementos de UI
        this.elements.weatherDescription.html(data.description);
        this.elements.temperature.html(`${data.temperature}°C`);
        this.elements.locationSection.html(`${data.city}, ${data.country}`);
        this.elements.feelsLike.html(`Sensación térmica: ${data.feelsLike}°C`);
        this.elements.humidity.html(`Humedad: ${data.humidity}%`);
        this.elements.windSpeed.html(`Viento: ${data.windSpeed} km/h`);

        // Agregar timestamp de última actualización
        const timestamp = new Date();
        const timeString = `${timestamp.getHours()}:${String(timestamp.getMinutes()).padStart(2, '0')}`;

        // Oculta el indicador de carga
        this.elements.loadingIndicator.fadeOut();
    }

    updateForecastUI(forecastData) {
        // Actualiza el título para reflejar "Próximos 3 días"
        
        // Crea el contenedor de filas si no existe
        const forecastRow = $('.forecast-row');
        forecastRow.empty();
    
        // Crea elementos para cada día del pronóstico
        forecastData.forEach(day => {
            const dayOfWeek = DIAS[day.date.getDay()];
            const forecastItem = `
                <div class="forecast-item">
                    <i class="weather-icon bi ${WEATHER_ICONS[day.icon] || 'bi-question-circle'}"></i>
                    <p class="day">${dayOfWeek}</p>
                    <p class="temperature">${day.temp}°C</p>
                </div>
            `;
    
            forecastRow.append(forecastItem);
        });
    }

    // Añade esta función a tu clase WeatherApp para procesar datos horarios
    async getHourlyForecast(lat, lon) {
        try {
            const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
            const response = await axios.get(url);
            const data = response.data;
            
            // Procesar datos para organizar por día
            const hourlyByDay = this.processHourlyForecastData(data.list);
            
            // Actualizar la UI con los datos horarios
            this.updateHourlyForecastUI(hourlyByDay);
            
        } catch (error) {
            console.error("Error getting hourly forecast data:", error);
        }
    }

    processHourlyForecastData(forecastList) {
        // Crea un objeto para almacenar pronósticos por día
        const hourlyByDay = {};
        
        forecastList.forEach(item => {
            const date = new Date(item.dt * 1000);
            const dateStr = date.toISOString().split('T')[0]; // YYYY-MM-DD
            
            // Si este día no existe en nuestro objeto, lo inicializa
            if (!hourlyByDay[dateStr]) {
                hourlyByDay[dateStr] = {
                    date: date,
                    dayName: DIAS_CORTO[date.getDay()],
                    hours: []
                };
            }
            
            // Añade la información horaria a este día
            hourlyByDay[dateStr].hours.push({
                hour: date.getHours(),
                icon: item.weather[0].icon,
                temp: item.main.temp.toFixed(0),
                description: item.weather[0].description,
                windSpeed: (item.wind.speed * 3.6).toFixed(1), // km/h
                humidity: item.main.humidity
            });
        });
        
        return Object.values(hourlyByDay).slice(1, 7);
    }

    updateHourlyForecastUI(hourlyByDay) {
        // Para cada día en el pronóstico diario
        hourlyByDay.forEach((day, index) => {
            // Crear contenedor para horas si no existe
            const dayElement = $(this.elements.forecastContainer.children()[index]);
            
            // Añade un botón o icono para expandir/colapsar
            if (!dayElement.find('.hourly-toggle').length) {
                dayElement.append(`
                    <div class="hourly-toggle bi bi-chevron-down" data-date="${day.date.toISOString().split('T')[0]}"></div>
                    <div class="hourly-container" id="hourly-${day.date.toISOString().split('T')[0]}" style="display: none;"></div>
                `);
            }
            
            // Rellenar el contenedor de horas
            const hourlyContainer = dayElement.find('.hourly-container');
            hourlyContainer.empty();
            
            // Añadir cada hora
            day.hours.forEach(hour => {
                hourlyContainer.append(`
                    <div class="hourly-item">
                        <span class="hourly-time ">${hour.hour}:00</span>
                        <i class="hourly-icon bi ${WEATHER_ICONS[hour.icon]}"></i>
                        <span class="hourly-temp">${hour.temp}°C</span>
                        <span class="hourly-wind"><i class="bi bi-wind"></i> ${hour.windSpeed} km/h</span>
                        <span class="hourly-humidity"><i class="bi bi-droplet"></i> ${hour.humidity}%</span>
                    </div>
                `);
            });
        });
        
        // Evento para mostrar/ocultar el pronóstico por horas
        $('.hourly-toggle').off('click').on('click', function() {
            const dateStr = $(this).data('date');
            const hourlyContainer = $(`#hourly-${dateStr}`);
            
            hourlyContainer.slideToggle(300);
            $(this).toggleClass('bi-chevron-down bi-chevron-up');
        });
    }
    
    // Método de depuración
    debug(message) {
        console.log(`[WeatherApp Debug] ${message}`);
    }

    // Obtiene y muestra el pronóstico horario de hoy
    async getTodayHourlyForecast(lat, lon) {
        this.debug('Iniciando getTodayHourlyForecast');
        try {
            const url = `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&appid=${CONFIG.API_KEY}&units=${CONFIG.WEATHER_UNITS}&lang=${CONFIG.WEATHER_LANG}`;
            this.debug(`Fetching URL: ${url}`);
            
            const response = await axios.get(url);
            this.debug('Datos recibidos correctamente');
            
            const data = response.data;
            if (!data || !data.list || !Array.isArray(data.list)) {
                throw new Error("Formato de datos inválido");
            }
            
            // Extraer pronóstico de las próximas horas
            const todayHourlyForecast = this.processTodayHourlyForecast(data.list);
            this.debug(`Procesados ${todayHourlyForecast.length} pronósticos horarios`);
            
            // Actualizar la UI con los datos horarios
            this.updateHourlyForecastDisplay(todayHourlyForecast);
            
        } catch (error) {
            console.error("Error getting today's hourly forecast data:", error);
            this.debug(`Error: ${error.message}`);
        }
    }

    // Procesamiento de datos horarios
    processTodayHourlyForecast(forecastList) {
        const hourlyData = [];
        const now = new Date();
        
        forecastList.forEach(item => {
            const forecastDate = new Date(item.dt * 1000);
            
            // Solo incluye pronósticos futuros (próximas 24 horas)
            if (forecastDate > now && (forecastDate - now) < 24 * 60 * 60 * 1000) {
                hourlyData.push({
                    hour: forecastDate.getHours(),
                    time: `${forecastDate.getHours()}:00`,
                    icon: item.weather[0].icon,
                    temp: item.main.temp.toFixed(0),
                    description: item.weather[0].description,
                    windSpeed: (item.wind.speed * 3.6).toFixed(1), // km/h
                    humidity: item.main.humidity
                });
            }
        });
        
        // Limitar a solo las próximas 3 horas o menos si no hay suficientes datos
        return hourlyData.slice(0, 3);
    }

    // Método para mostrar el pronóstico horario
    updateHourlyForecastDisplay(hourlyData) {
        this.debug('Iniciando updateHourlyForecastDisplay');
        
        // Encuentra el contenedor existente en el DOM
        const hourlyContainer = $('.hourly-forecast');
        
        // Verifica si el contenedor existe
        if (hourlyContainer.length === 0) {
            this.debug('ERROR: No se encontró el contenedor hourly-forecast');
            return;
        }
        
        this.debug(`Actualizando ${hourlyData.length} items en el pronóstico horario`);
        
        // Limpia el contenedor
        hourlyContainer.empty();
        
        // Si no hay datos para mostrar
        if (hourlyData.length === 0) {
            hourlyContainer.append('<p>No hay datos disponibles para las próximas horas</p>');
            return;
        }
        
        // Crea un elemento para cada hora pronosticada
        hourlyData.forEach(hourData => {
            const forecastItem = `
                <div class="forecast-item">
                    <p class="time">${hourData.time}</p>
                    <i class="weather-icon bi ${WEATHER_ICONS[hourData.icon] || 'bi-question-circle'}"></i>
                    <p class="temperature">${hourData.temp}°C</p>
                    <p class="status">${hourData.description}</p>
                </div>
            `;
            
            hourlyContainer.append(forecastItem);
        });
        
        this.debug('Actualización del pronóstico horario completada');
    }
}

// Inicia la app weather cuando el documento esté listo
$(function () {
    const weatherApp = new WeatherApp();
});