import React, { useState } from 'react';
import axios from 'axios';
import "../../css/AudioEnchancer.css";

function AudioEnchancerJSX() {
    const [file, setfile] = useState("");
    const [EnchancedAudio,setEnchancedAudio] = useState(""); 
    const [loading,setLoading] = useState(false); 

        setLoading(true); // set loading to true when feedback submission starts


        const processAudio = () => {

            axios.post('http://localhost:5000/UploadAudio', {
                response: AIresponse,
                score: score
        }).then(response => {
            console.log(response.data);
            setLoading(false); // Optionally reset score after submission
        }).catch(error => {
            console.error('Error submitting feedback:', error);
        }).finally( () => {
            setLoading(false);  // Reset loading state when feedback submission completes
        })
    };
    
return (
    <div className='Audio-Container'>

    </div>
)
}

export default AudioEnchancerJSX;