const chatBox = document.getElementById("chat-box");
const userMessageInput = document.getElementById("user-message");
const sendButton = document.getElementById("send-button");

function addMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(`${sender}-message`);
    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);
}

addMessage("Hello, I'm your friendly bot! How can I help you?", "bot");

// Function to handle sending user messages
function sendMessage() {
    const userMessage = userMessageInput.value;
    if (userMessage.trim() !== "") {
        addMessage(userMessage, "user");
        userMessageInput.value = "";
        fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: `question=${encodeURIComponent(userMessage)}`,
        })
        .then(response => response.json())
        .then(data => {
            addMessage(data.answer, "bot");
        })
        .catch(error => {
            console.error("Error:", error);
        });
    }
}


// Listen for Enter key press in the user input field
userMessageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent the default Enter key behavior (e.g., line break)
        sendButton.click(); // Trigger a click event on the Send button
    }
});

// Listen for click event on the Send button
sendButton.addEventListener("click", sendMessage);
