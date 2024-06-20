import '../index.css'
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
    return (
        <div className="w-60 bg-white shadow-md px-5 py-7">
            <h1 className="text-xl font-semibold mb-10">Optimization</h1>
            <ul className="space-y-4">
                <li>
                    <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                    >
                        Toolbox
                    </button>
                </li>
                <li>
                    <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                    >
                        Results
                    </button>
                </li>
                <li>
                    <button
                        className="text-gray-700 hover:text-gray-900 bg-transparent border-none cursor-pointer"
                    >
                        Track progress
                    </button>
                </li>
            </ul>
        </div>
    );
}

export default Sidebar;