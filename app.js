const chatButton = document.querySelector('.chatbox__button button');
const chatBox = document.querySelector('.chatbox__support');

chatButton.addEventListener('click', () => {
    chatBox.classList.add('chatbox--active');
});

class Chatbox {
    constructor() {
        this.args = {
            sendButton: document.querySelector('.send__button'),
            messagesContainer: document.querySelector('.chatbox__messages')
        };

        this.messages = [];
    }

    display() {
        const { sendButton } = this.args;

        sendButton.addEventListener('click', () => this.onSendButton());

        const inputNode = chatBox.querySelector('input');
        inputNode.addEventListener('keyup', ({ key }) => {
            if (key === 'Enter') {
                this.onSendButton();
            }
        });
    }

    renderMessage(message) {
        const { messagesContainer } = this.args;
        const messageDiv = document.createElement('div');
        messageDiv.textContent = `${message.name}: ${message.message}`;
        messagesContainer.appendChild(messageDiv);
    }

    onSendButton() {
        const inputField = chatBox.querySelector('input');
        const messageText = inputField.value;

        if (messageText === '') {
            return;
        }

        const messageObj = { name: 'User', message: messageText };
        this.messages.push(messageObj);
        this.renderMessage(messageObj);

        // Clear the input field after sending the message
        inputField.value = '';
    }
}

const chatbox = new Chatbox();
chatbox.display();
