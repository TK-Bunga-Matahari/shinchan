import '../index.css';
import { useNavigate } from 'react-router-dom';


const Sidebar = () => {
    const navigate = useNavigate();

    const handleMenu = () => {
        navigate("/menu");
    }

    const handleResult = () => {
        navigate("/menu/result");
    }

    const handleProgress = () => {
        navigate("/menu/progress");
    }

    const handleHome = () => {
        navigate("/");
    }

    return (
        <div className="w-60 bg-white shadow-md px-5 py-7">
                <button className="text-xl font-semibold mb-10" onClick={handleHome}>Optimization</button>
                <ul className="space-y-4">
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleMenu}
                        >
                        Toolbox
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleResult}
                        >
                        Results
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        onClick={handleProgress}
                        >
                            Track progress
                        </button>
                    </li>
                    <li>
                        <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                        // onClick={handleLogout}
                        >
                            Logout
                        </button>
                    </li>
                </ul>
            </div>
    );
}

export default Sidebar;