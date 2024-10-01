import React, { useState } from 'react';
import axios from 'axios';
import "../../css/ChatGpt.css";

function Chatbot() {
    const [input, setInput] = useState("");
    const [AIresponse, setAIResponse] = useState("");
    const [score, setScore] = useState(null);  // Initialize score state
    const [feedbackSubmitted,setFeedbackSubmitted] = useState(false);  //tracking feedback
    const [loading,setLoading] = useState(false); // track request loading state

    const AskAi = async () => {
        if (input.trim() === "") return;
    
        setLoading(true);// set loading to true/in proccess

        const requestBody = {
            message: input
        };
    
        try {
            const response = await fetch('http://localhost:5000/chatbot/chat', {
                method: 'POST',
                headers: {
                    'Content-type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
    
            const data = await response.json();
            setAIResponse(data.response);
            console.log('AI Response:', data.response);
            setFeedbackSubmitted(false)
        } catch (error) {
            console.error('Error:', error);
            alert('Error communicating with the server. Please try again.');
        } finally{
            setLoading(false); //reset loading once the request is complete
        }
    };
    
    const handleFeedback = (scoreValue) => {  
        setScore(scoreValue);
        console.log('Feedback score set to:', scoreValue); // Log the feedback score
    };

    const handleSubmit = () => {
        if (score === null) {
            alert('Please provide feedback before submitting.');
            return;
        }

        setLoading(true); // set loading to true when feedback submission starts

        axios.post('http://localhost:5000/chatbot/feedback', {
            response: AIresponse,
            score: score
        }).then(response => {
            console.log('Feedback submitted successfully:', response.data);
            setFeedbackSubmitted(true); // updates feedback status
            setScore(null); // Optionally reset score after submission
        }).catch(error => {
            console.error('Error submitting feedback:', error);
        }).finally( () => {
            setLoading(false);  // Reset loading state when feedback submission completes
        })
    };

    return (
        <div className='PageContainer'>
            <div className='GPTBackground'>
                <h1>Chat with AI Personal Trainer</h1>
                <div className='userinputcontainer'>
                <div>
                    <p className='Chatbot-P'>{AIresponse}</p> 
                    <p className='User-P'>{input}</p>
                </div>
                    <input placeholder='Ask AI' className='ChatBox' type='text' value={input} onChange={(e) => setInput(e.target.value)}/>
                </div>
                    <button  className='sendmessage' onClick={AskAi} disabled={loading}>{loading ? 'sending...' : 'Send Message'}</button>
                <div className='Feedback-Container'>
                    <button onClick={() => handleFeedback(1)} disabled={feedbackSubmitted || loading}>👍 Positive</button>
                    <button onClick={() => handleFeedback(-1)} disabled={feedbackSubmitted || loading}>👎 Negative</button>
                    <button  onClick={handleSubmit} disabled={loading}> {loading ? 'submitting...' : 'submit feedback'}</button>
                </div>
                <div>
                </div>
            </div>
        </div>
    );
}

export default Chatbot;