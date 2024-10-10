import React, { useState } from "react";
import axios from "axios";
import "../../css/ChatGpt.css";

function Chatbot() {
  const [input, setInput] = useState("");
  const [AIresponse, setAIResponse] = useState("");
  const [score, setScore] = useState(null); // Initialize score state
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false); //tracking feedback
  const [loading, setLoading] = useState(false); // track request loading state
  const [inputsent,setInputsent] = useState(false); // track if the input has been sent
  const AskAi = async () => {
    if (input.trim() === "") return;

    setLoading(true); // set loading to true/in proccess

    const requestBody = {
      message: input
    };

    try {
      const response = await fetch("http://localhost:5000//chatbot/chat", {
        method: "POST",
        headers: {
          "Content-type": "application/json"
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setAIResponse(data.response);
      console.log("AI Response:", data.response);
      setInputsent(true);
      setFeedbackSubmitted(false);
    } catch (error) {
        console.error('Error:', error);
        if (error.response) {
            console.error('Response data:', error.response.data);
            console.error('Response status:', error.response.status);
        }
        alert('Error communicating with the server. Please try again.');
    
    } finally {
      setLoading(false); 
    }
  };

  const handleFeedback = scoreValue => {
    setScore(scoreValue);
    console.log("Feedback score set to:", scoreValue); 
  };

  const handleSubmit = () => {
    if (score === null) {
      alert("Please provide feedback before submitting.");
      return;
    }

    setLoading(true);

    axios
      .post("http://localhost:5000/chatbot/feedback", {
        response: AIresponse,
        score: score
      })
      .then(response => {
        console.log("Feedback submitted successfully:", response.data);
        setFeedbackSubmitted(true); 
        setScore(null); 
      })
      .catch(error => {
        console.error("Error submitting feedback:", error);
      })
      .finally(() => {
        setLoading(false); 
      });
  };

  return (
    <div className="PageContainer">
      <div className="GPTBackground">
        <h1>Chat with AI Personal Trainer</h1>
        <div className="userinputcontainer">
          <div>
          { inputsent && ( <>
            <p className="Chatbot-P">AI: {AIresponse}</p>
            <p className="User-P">User:  {input} </p>
            </>
          )}
          </div>
   
          <input  placeholder="Ask AI" className="ChatBox"  type="text"  value={input}  onChange={e => setInput(e.target.value)}   />
      
        </div>
        <button className="sendmessage" onClick={AskAi} disabled={loading}>
          {loading ? "sending..." : "Send Message"}
        </button>
        <div className="Feedback-Container">
          <button
            onClick={() => handleFeedback(1)}
            disabled={feedbackSubmitted || loading}
          >
            👍 Positive
          </button>
          <button
            onClick={() => handleFeedback(-1)}
            disabled={feedbackSubmitted || loading}
          >
            👎 Negative
          </button>
          <button onClick={handleSubmit} disabled={loading}>
            {" "}{loading ? "submitting..." : "submit feedback"}
          </button>
        </div>
        <div />
      </div>
    </div>
  );
}

export default Chatbot;
