import { useAuth } from "../Components/Authanciation/AuthProvider";
import DropdownMenu from "../Components/DropDownController";
function Dashboard() {
  const { user, logout } = useAuth();
  return (
    <div>
      <button onClick={logout}>Logout</button>
      <h1>
        Welcome {user}
      </h1>
      <DropdownMenu />
      <div />
      <div className="scroll-container">
        <div className="scroll-page - QuoteText" id="page-1">
          <p />
        </div>
        <div className="scroll-page   - QuoteText" id="page-2">
          <p />
        </div>
        <div className="scroll-page   - QuoteText" id="page-3">
          <p />
        </div>
      </div>
    </div>
  );
}
export default Dashboard;
