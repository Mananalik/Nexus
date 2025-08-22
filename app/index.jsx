import { useRouter } from "expo-router";
import { Image, ScrollView, Text, TouchableOpacity, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import logo from "../assets/images/NexusLogo.png";

export default function Index() {
  const router = useRouter();
  return (
    <SafeAreaView className="bg-[#030303]">
      <ScrollView contentContainerStyle={{ height: "100%" }}>
        <View className="flex-1 flex flex-col p-8">
          <View className="flex flex-col items-center mb-28">
            <Image source={logo} style={{ height: 250, width: 250 }} />
            <Text className="text-[#F1EFEC] text-5xl font-bold tracking-wider">
              Nexus
            </Text>
            <Text className="text-[#D4C9BE] text-lg mt-2">
              Your finances, connected.
            </Text>
          </View>

          <View className="w-full max-w-sm">
            <TouchableOpacity
              className="bg-[#123458] flex justify-center items-center w-full py-4 rounded-xl mb-4 text-[#F1EFEC] text-lg font-bold transition-opacity hover:opacity-90"
              onPress={() => router.push("/signup")}
            >
              <Text className="text-[#F1EFEC] text-lg font-bold">Sign Up</Text>
            </TouchableOpacity>

            <TouchableOpacity
              className="flex justify-center items-center bg-transparent border border-[#123458] w-full py-4 rounded-xl  transition-colors hover:bg-[#123458]/20"
              onPress={() => router.push("/signin")}
            >
              <Text className="text-[#F1EFEC] text-lg font-bold">Sign In</Text>
            </TouchableOpacity>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
