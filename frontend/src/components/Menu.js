import '../index.css';
import { Link } from 'react-router-dom';
import { useNavigate } from 'react-router-dom';
import image from '../icon/task_2098402.svg';
import image2 from '../icon/profit_7172432.svg';
import { auth } from '../firebase-config';
import { signOut } from 'firebase/auth';
import Sidebar from '../layouts/sidebar';

const Menu = () => {

    const navigate = useNavigate();

    const handleTaskAssignment = () => {
        navigate("/menu/taskassignments");
    }

    const handlePortfolioAllocation = () => {
        navigate("/menu/portfolioallocation");
    }

    return (
        <div className="bg-gray-100 min-h-screen flex">
            <Sidebar />

            {/* Main Content */}
            <div className="flex-grow p-10">
                <div className="max-w-4xl mx-auto">
                <h2 className="text-2xl font-bold mb-6">Optimization Toolbox</h2>
                <div className="grid grid-cols-2 gap-4">
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex flex-col items-center justify-center" onClick={ handleTaskAssignment }>
                        <img src={image} className='taskImage mb-2' alt="Task Assignment" />
                        Task Assignment
                    </button>
                    <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded flex flex-col items-center justify-center">
                        <img src={image2} className='taskImage mb-2' alt="Portfolio Allocation" />
                        Portfolio Stock Allocation
                    </button>
                </div>
                </div>
            </div>
        </div>
    );
}

export default Menu;
