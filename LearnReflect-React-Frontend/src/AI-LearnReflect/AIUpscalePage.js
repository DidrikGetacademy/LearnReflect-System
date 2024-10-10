import React from "react";
import { useNavigate } from "react-router-dom";
import '../css/AiUpscalePage.css';
function AIUpscalePage() {
  const navigate = useNavigate();
  return (
    <div>
      <h1>AI Upscale Page</h1>
      <div className="AIupscale-Container">
        <div>
          <img alt="AudioEnchancer" />
          <button onClick={() => navigate('/AudioRender')}>Audio Enchancer</button>
          <p>
      
          </p>
        </div>

        <div>
          <img alt="VideoEnchancer" />
          <button onClick={() => navigate('/')}>Video Enchancer</button>
          <p>
       
          </p>
        </div>

        <div>
          <img alt="Chatbot" />
          <button onClick={() => navigate('/Chatbot')}>AI Chatbot</button>
          <p>
      
          </p>
        </div>
      </div>
    </div>
  );
}
export default AIUpscalePage;
