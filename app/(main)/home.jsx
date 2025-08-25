import { Text } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { View } from "react-native-web";

const Home = () => {
  return (
    <SafeAreaView className="bg-[#030303]">
      <View className="flex-col flex-1 justify-center items-center">
        <Text className="text-[#F1EFEC] text-4xl">Home Page</Text>
      </View>
    </SafeAreaView>
  );
};

export default Home;
