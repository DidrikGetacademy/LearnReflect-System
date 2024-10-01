import {useState, useEffect} from 'react';
function Timer(){
const [count,SetCount] = useState({
days: 7,
hours: 0,
minutes: 0,
seconds: 0
});
useEffect(()=>{
const interval = setInterval(() => {
    SetCount(prevCount => {
    if(prevCount.seconds === 0){
    if(prevCount.minutes === 0){
    if(prevCount.hours === 0){
    if(prevCount.days === 0){
    clearInterval(interval);
    return prevCount;
    }else {
        return {
            days: prevCount.days-1,
            hours:23,
            minutes: 59,
            seconds:59,    
        };
      }
    }
    else {
        return {
        ...prevCount,
        hours: prevCount.hours -1,
        minutes: 59,
        seconds: 59,
        };
      }  
    }
    else {
        return {
        ...prevCount,
        minutes: prevCount.minutes -1,
        seconds: 59,
        };
      }
    }  
   else {
       return {
    ...prevCount,
      seconds: prevCount.seconds - 1,
  };}});
}, 1000);


return ()=> clearInterval(interval);
},[]);


return (
<div>
    <h1>Countdown</h1>
    <h4>{count.seconds} Seconds</h4>
    <h4>{count.minutes} Minutes</h4>    
    <h4>{count.hours} Hours</h4>
    <h4>{count.days} days </h4>
  
</div>
);
}

export default Timer;