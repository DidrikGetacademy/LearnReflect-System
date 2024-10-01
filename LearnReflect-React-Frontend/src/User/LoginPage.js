import React, { useState } from "react";
import axios from "axios";
import "../css/Homepage.css";
import DropdownMenu from "../Components/DropDownController";
import Bcomponent from "../Components/BComponent";
import LComponent from "../Components/LogoComponent";
import { useAuth } from "../Components/Authanciation/AuthProvider";
import { EmptyInput } from "../Components/Validation";
function LoginPage() {
  const { loginAction } = useAuth();
  const [ErrorMessage, setErrorMessage] = useState("");
  const [values, setValues] = useState({
    name: "",
    password: ""
  });



  const message = {
    IncorrectUsername: "Incorrect Username",
    EmptyField: "Fields can't be empty",
    WrongPassword: "Wrong Password",
    ServerError: "Server Error",
    incorrectCredential: "Please enter a valid email and password."
  };

  const handleInput = event => {
    setValues(prev => ({...prev,[event.target.name]: event.target.value }));
  };


  const handleSubmit = async event => {
    event.preventDefault();
    if (!EmptyInput(values.name,values.password)) {
      try {
        const response = await axios.post( "http://localhost:8081/login",  values);
        const responseData = response.data;
        if (responseData.success) {
          loginAction(responseData);
        } else if (responseData === "Incorrect Username or Email") {
          setErrorMessage(message.IncorrectUsername);
        } else if (responseData === "Incorrect password") {
          setErrorMessage(message.WrongPassword);
        } else if (responseData === "No user exsist") {
          setErrorMessage(message.IncorrectUsername);
        } else {
          setErrorMessage(message.ServerError);
        }
      } catch (error) {
        console.error("Error:", error);
        alert("Error occurred while processing your request");
      }
    } else return;
  };

  return (
    <div>
      <h1>Login</h1>
      <LComponent />
      <Bcomponent />
      <DropdownMenu />
      <div>
        <div className="logincontainer">
          <form onSubmit={handleSubmit} className="loginform">
            <label className="ErrorLabel">
              {ErrorMessage}
            </label>
            <br />
            <input
              onChange={handleInput}
              name="name"
              placeholder="Username or Email"
            />
            <input
              onChange={handleInput}
              name="password"
              placeholder="Password"
              type="password"
            />
            <button type="submit" className="LoginButton">
              Login
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
export default LoginPage;
