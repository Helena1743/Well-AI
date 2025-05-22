// Function to check if required fields are filled
function checkRequiredFields() {
    const requiredInputs = document.querySelectorAll('.required-input');
    let allSelected = true;

    requiredInputs.forEach(input => {
        if (input.type === 'radio') {
            const name = input.name;
            const checkedRadio = document.querySelector(`input[name="${name}"]:checked`);
            if (!checkedRadio) {
                allSelected = false;
            }
        } else if (input.type === 'text' && input.value.trim() === '') {
            allSelected = false;
        }
    });

    return allSelected;
}

// Function to show the loading bar and scroll to the bottom
function showLoadingBar() {
    // Check if all required fields are filled
    if (checkRequiredFields()) {
        // Show the loading bar
        document.getElementById("loading-bar").style.display = "block";
        
        // Scroll to the bottom to show the loading bar
        setTimeout(function() {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }, 100);
        
        // Allow form submission
        document.getElementById('submitButton').disabled = false;  // Ensure the button is enabled
        document.getElementById('prediction-form').submit();  // Submit the form
    } else {
        // Display an error message if not all fields are filled
        const errorMessage = document.getElementById('error-message');
        errorMessage.style.display = 'block';
    }
}

// Show or hide the error message when the form is submitted
document.getElementById('submitButton').addEventListener('click', function(event) {
    const errorMessage = document.getElementById('error-message');

    // Check if all required fields are filled
    const formIsValid = checkRequiredFields();

    if (!formIsValid) {
        // Prevent form submission if not all fields are filled
        event.preventDefault();

        // Show the error message
        errorMessage.style.display = 'block';
    } else {
        // Hide the error message if all fields are filled
        errorMessage.style.display = 'none';

        // Call function to show the loading bar and submit the form
        showLoadingBar();
    }
});

// Call the function on page load to set initial button state
window.onload = function() {
    checkRequiredFields(); // Check if fields are filled when the page loads
};


window.onload = function() {
    const resultsSection = document.getElementById('results-section');
    
    // Check if results are displayed
    if (resultsSection && resultsSection.innerHTML.trim() !== "") {
        // Scroll to the results section
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
};