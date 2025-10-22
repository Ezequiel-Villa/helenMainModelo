
$(function() {
const clock = $('.clock'); // select the clock class <p> tag
const dateSection = $('.date-section'); // select the date class <p> tag
const locationSection = $('.location'); // select the location class <p> tag
const weatherIcon = $('.weather-icon'); // select the weather-icon class <i> tag
const temperature = $('.temperature'); // select the temperature class <p> tag
const weatherDescription = $('.status'); // select the weather-description class <p> tag

const activeSensor = $('.active-sensor');

// Function to get the current time
const getTime = () => {
    const date = new Date(); // create a new date object
    const hours = date.getHours(); // get the hours
    const minutes = date.getMinutes(); // get the minutes
    const seconds = date.getSeconds(); // get the seconds
    const day = date.getDate(); // get the day
    const month = date.getMonth(); // get the month
    const year = date.getFullYear(); // get the year
    const dayOfWeek = date.getDay(); // get the day of the week

    // Array of days of the week
    const days = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado'];

    // Array of months
    const months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];

    // Display the time
    let formatedHours = hours > 12 ? (hours-12 > 10 ? `0${hours}`:hours) : (hours < 10 ? `0${hours}`:hours);
    let formatedMinus = minutes < 10 ? `0${minutes}` : minutes;
    let formatedSeconds = seconds < 10 ? `0${seconds}` : seconds;
    let timeString = `${formatedHours}:${formatedMinus}:${formatedSeconds}`;
    clock.html(timeString);

    // Display the date
    let dateString = `${days[dayOfWeek]}, ${day} de ${months[month]} de ${year}`;
    dateSection.html(dateString);


    setTimeout(getTime, 1000);
}

const getLocation = () => {
    const lat = 32.43347;
    const lon = -116.67447;
    const city = 'Tijuana';
    const country = 'MX';

    let locationString = `${city}, ${country}`;
    locationSection.html(locationString);
};

const getWeather = async() => {
    const key = '6266f75957014a7de4ae0ded34d1e7cc';
    const lat = 32.43347;
    const lon = -116.67447;
    const url = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${key}&units=metric&lang=es`;

    const response = await axios.get(url);
    const data = response.data;

    let watherData = {
        icon: data.weather[0].icon,
        temperature: data.main.temp.toFixed(0),
        description: data.weather[0].description
    };

    let mappingIcons = {
        //bootsrap icons
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

    weatherIcon.addClass(mappingIcons[watherData.icon]);
    weatherDescription.html(watherData.description);
    temperature.html(`${watherData.temperature}°C`);



    setTimeout(getWeather, 1000*60*10);// 10 minutes
};

activeSensor.on('click', function() {
    alert('Sensor activado');
});


getTime();
getLocation();
getWeather();
});

