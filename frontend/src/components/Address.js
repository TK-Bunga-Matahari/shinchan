import '../index.css';
import { useNavigate } from 'react-router-dom';

const Address = () => {
    const navigate = useNavigate();

    const handleSignUp = () => {
        navigate("/");
    }

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-100">
            <div className="text-center">
                <h1 className="text-4xl font-bold mb-4">Optimization Project</h1>
                <p className="text-lg mb-8">TK Bunga Matahari</p>
                <div className="flex flex-col items-center">
                    <input
                        type="text"
                        className="w-80 p-2 border border-gray-400 rounded shadow"
                        placeholder="Enter your address"
                    />
                    <button
                        className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold py-2 px-4 border border-gray-400 rounded shadow mt-4"
                        onClick={() => {
                            // Handle address submit logic here
                            console.log("Address submitted");
                        }}
                    >
                        Submit
                    </button>
                    <p className="text-sm mt-4"><a href="" onClick={handleSignUp}>Sign Up</a></p>
                </div>
            </div>
        </div>
    );
}

export default Address;