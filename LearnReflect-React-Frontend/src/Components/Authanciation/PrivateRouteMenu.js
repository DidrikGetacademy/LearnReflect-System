
import { useAuth } from './AuthProvider'; 
const PrivateRouteMenu = ({ children }) => {
  const { isAuthenticated } = useAuth(); 
 return children(isAuthenticated);
};
export default PrivateRouteMenu;
