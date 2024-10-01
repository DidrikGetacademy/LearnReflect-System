//Authprovider
import React, { createContext, useContext, useState } from "react";
import { useNavigate } from "react-router-dom";

const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem("site") || "");
  const isAuthenticated = !!user;
  const navigate = useNavigate();

  const loginAction = async data => {
    try {
      if (data && data.username) {
        setUser(data.username);
        setToken(data.token);
        localStorage.setItem("site", data.token);
        navigate("/Dashboard");
      } else {
      }
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const logout = () => {
    setUser(null);
    setToken("");
    localStorage.removeItem("site");
    navigate("/LR");
  };

  return (
    <AuthContext.Provider
      value={{ isAuthenticated, token, user, loginAction, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export default AuthProvider;
export const useAuth = () => useContext(AuthContext);
