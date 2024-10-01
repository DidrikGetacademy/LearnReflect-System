import React from 'react';
import DropdownMenu from '../Components/DropDownController';
import Bcomponent from '../Components/BComponent';
import LComponent from '../Components/LogoComponent';
function AboutPage(){
  
return(
    <div>
      <h1 className='hh1'>About LearnReflect</h1>
         <LComponent/>
         <Bcomponent/>
         <DropdownMenu /> {}
         <div>
          <p className='about-text'></p>
         </div>
</div>
);
}

export default AboutPage;

